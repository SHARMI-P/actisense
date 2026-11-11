"""Train and evaluate the activity model on <data>/<activity>/*.npy windows.

v4 (as before):
  python train.py --data data data_penn --test-pattern "person(02|03|05|06|07|08|09|10|22)_|pennTest_" \
                  --features xyz --augment --tag v4
v5 (optimised): dynamic activities from public datasets only, motion features, stronger augmentation and model:
  python train.py --data data data_penn --test-pattern "..." --features xyz --augment \
                  --exclude-own --drop-classes standing,sitting,lying \
                  --velocity --mirror --speed --model tcn --seeds 3 --tag v5

Data split (three parts, never mixed):
  test       - dataset files matching --test-pattern (held-out people / videos) + 20% of own recordings
  validation - 15% of the remaining clips; used ONLY to pick the best epoch and the best seed
  train      - everything else
Windows cut from the same video always stay in the same part, so nothing leaks between them.

Outputs: results_<tag>.txt, confusion_matrix_<tag>.png, har_model_<tag>.pt, and one row per run appended to
ablation.csv (so several settings can be compared in one table).
"""
import argparse
import csv
import glob
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from common import LR_PERM, add_velocity, make_model, select_features, MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", default=["data"], help="one or more folders of <activity>/*.npy windows")
ap.add_argument("--dataset-pattern", default=r"^(person\d+_|penn(Train|Test)_)",
                help="file names from public datasets (KTH, Penn Action); everything else = own recordings")
ap.add_argument("--test-pattern", default="",
                help="regex on file names; matching dataset files form the test set (held-out people)")
ap.add_argument("--exclude-own", action="store_true",
                help="leave out own webcam recordings (one person, one room: a source of bias)")
ap.add_argument("--drop-classes", default="", help="comma list of activities to leave out, e.g. standing,sitting,lying")
ap.add_argument("--merge", default="", help="rename classes: old=new,... (e.g. jogging=running)")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--patience", type=int, default=10, help="stop after this many epochs without improvement")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weighting", choices=["inverse", "sqrt", "none"], default="sqrt",
                help="class weights: inverse = strong, sqrt = softer (default), none")
ap.add_argument("--features", choices=["all", "xyz"], default="all",
                help="xyz = drop MediaPipe visibility scores (removes a 'which camera' shortcut)")
ap.add_argument("--velocity", action="store_true", help="add frame-to-frame joint movement as features")
ap.add_argument("--augment", action="store_true", help="random rotation (+-20 deg), scale and keypoint noise")
ap.add_argument("--mirror", action="store_true", help="random left-right mirroring (left/right-handed people)")
ap.add_argument("--speed", action="store_true", help="random speed change 0.75x-1.33x (faster/slower people)")
ap.add_argument("--model", choices=sorted(MODELS), default="lstm", help="lstm (v1-v4), bilstm (+attention), tcn")
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--seeds", type=int, default=1, help="train with this many random seeds; report mean and spread")
ap.add_argument("--tag", default="", help="suffix for output files, e.g. v5 -> results_v5.txt")
ap.add_argument("--ablation-csv", default="ablation.csv")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBJECT_FILE = re.compile(args.dataset_pattern)
drop = {c.strip() for c in args.drop_classes.split(",") if c.strip()}
merge = dict(kv.split("=", 1) for kv in args.merge.split(",") if "=" in kv)

# ---------------- load ----------------
files = []                                   # (path, class)
for root in args.data:
    for d in sorted(os.listdir(root)):
        for f in sorted(glob.glob(os.path.join(root, d, "*.npy"))):
            name = os.path.basename(f)
            if args.exclude_own and not SUBJECT_FILE.match(name):
                continue
            cls = merge.get(d, d)
            if cls not in drop:
                files.append((f, cls))
actions = sorted({c for _, c in files})
if len(actions) < 2:
    raise SystemExit(f"Need data for at least 2 activities in {args.data}")

X = np.stack([np.load(f) for f, _ in files]).astype(np.float32)
y = np.array([actions.index(c) for _, c in files], dtype=np.int64)   # int64: Windows defaults to int32
names = np.array([os.path.basename(f) for f, _ in files])
groups = np.array([c + "/" + re.sub(r"_\d+\.npy$", "", os.path.basename(f)) for f, c in files])  # one per clip
is_dataset = np.array([bool(SUBJECT_FILE.match(n)) for n in names])
X = select_features(X, args.features)
RAW_NF = X.shape[-1]
print(f"Classes ({len(actions)}): {actions}")
print(f"Windows: {len(X)}  (dataset {is_dataset.sum()}, own recordings {(~is_dataset).sum()})  device: {device}")


def group_split(idx, test_size, seed=42):
    """Split indices so that all windows of one clip land on the same side."""
    if len(idx) == 0 or len(set(groups[idx])) < 2:
        return idx, idx[:0]
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                .split(idx, groups=groups[idx]))
    return idx[a], idx[b]


# ---------------- split: train / validation / test ----------------
all_idx = np.arange(len(X))
if args.test_pattern:
    ds_idx, own_idx = all_idx[is_dataset], all_idx[~is_dataset]
    in_test = np.array([bool(re.search(args.test_pattern, n)) for n in names[ds_idx]])
    if not in_test.any():
        raise SystemExit("--test-pattern matched no files")
    own_rest, own_test = group_split(own_idx, 0.2)
    rest_idx = np.concatenate([ds_idx[~in_test], own_rest])
    te_idx = np.concatenate([ds_idx[in_test], own_test])
else:
    rest_idx, te_idx = group_split(all_idx, 0.2)
tr_idx, va_idx = group_split(rest_idx, 0.15, seed=7)
print(f"Split: train {len(tr_idx)} / validation {len(va_idx)} / test {len(te_idx)} windows "
      f"({len(set(groups[te_idx]))} test clips)")
missing = [actions[c] for c in range(len(actions)) if c not in set(y[te_idx])]
if missing:
    print(f"WARNING: no test windows for {missing}")


# ---------------- features, augmentation, normalisation ----------------
def feats(x):
    """raw (B, T, RAW_NF) tensor -> model input (+ velocity)."""
    return add_velocity(x) if args.velocity else x


PERM = torch.tensor(LR_PERM)


def augment(xb):
    """Random camera- and person-like changes to a batch of raw skeleton windows (B, T, RAW_NF).
    The same change is applied to all frames of one window."""
    xb = xb.clone()
    B, T = xb.shape[:2]
    xyz = xb[..., :99].reshape(B, T, 33, 3)
    if args.mirror:                                   # left <-> right body sides, x flipped
        flip = torch.rand(B) < 0.5
        m = xyz[flip][:, :, PERM].clone()
        m[..., 0] = -m[..., 0]
        xyz[flip] = m
        if RAW_NF > 99:
            xb[flip, :, 99:] = xb[flip][:, :, 99:][:, :, PERM]
    if args.augment:
        theta = (torch.rand(B) * 2 - 1) * np.radians(20)        # turn left/right up to 20 degrees
        c, s = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
        x, z = xyz[..., 0].clone(), xyz[..., 2].clone()
        xyz[..., 0], xyz[..., 2] = c * x + s * z, -s * x + c * z
        xyz *= (0.85 + 0.3 * torch.rand(B))[:, None, None, None]  # body size / distance
        xyz += 0.02 * torch.randn_like(xyz)                       # keypoint noise
    xb[..., :99] = xyz.reshape(B, T, 99)
    if args.speed:                                    # resample time: 0.75x .. 1.33x speed
        f = torch.exp((torch.rand(B) * 2 - 1) * np.log(1.33))
        centre = (T - 1) / 2 + (torch.rand(B) * 2 - 1) * (T - 1) / 2 * (1 - torch.clamp(f, max=1.0))
        t = centre[:, None] + (torch.arange(T)[None] - (T - 1) / 2) * f[:, None]
        t = t.clamp(0, T - 1)
        lo = t.floor().long()
        hi = (lo + 1).clamp(max=T - 1)
        w = (t - lo)[..., None]
        gather = lambda i: torch.gather(xb, 1, i[..., None].expand(-1, -1, RAW_NF))
        xb = gather(lo) * (1 - w) + gather(hi) * w
    return xb


Xt = torch.from_numpy(X)
F_tr = feats(Xt[tr_idx])
NF = F_tr.shape[-1]
mean = F_tr.reshape(-1, NF).mean(0)
std = F_tr.reshape(-1, NF).std(0) + 1e-6
del F_tr
norm = lambda f: (f - mean) / std

counts = np.bincount(y[tr_idx], minlength=len(actions)).astype(np.float32)
weights = counts.sum() / (len(actions) * np.maximum(counts, 1))
weights = np.sqrt(weights) if args.weighting == "sqrt" else (np.ones_like(weights) if args.weighting == "none" else weights)
weights = weights / weights.mean()
print("Train windows per class:", dict(zip(actions, counts.astype(int).tolist())))


def predict_proba(model, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024):
            xb = norm(feats(Xt[idx[i:i + 1024]])).to(device)
            out.append(torch.softmax(model(xb), 1).cpu())
    return torch.cat(out).numpy()


def video_level(prob, idx):
    """Average the window probabilities of each test clip -> one prediction per clip."""
    g = groups[idx]
    clips = sorted(set(g))
    yt = np.array([y[idx][g == c][0] for c in clips])
    yp = np.array([prob[g == c].mean(0).argmax() for c in clips])
    return yt, yp


# ---------------- train (one run per seed), best epoch chosen on VALIDATION ----------------
def train_once(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(args.model, NF, len(actions), args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device), label_smoothing=0.05)
    dl = DataLoader(TensorDataset(Xt[tr_idx], torch.from_numpy(y[tr_idx])), batch_size=args.batch, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for xb, yb in dl:
            if args.augment or args.mirror or args.speed:
                xb = augment(xb)
            xb, yb = norm(feats(xb)).to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        va_f1 = f1_score(y[va_idx], predict_proba(model, va_idx).argmax(1), average="macro") if len(va_idx) else 0.0
        sched.step(va_f1)
        if va_f1 > best_f1:
            best_f1, best_epoch, bad = va_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  seed {seed} epoch {epoch:3d}  loss {total / len(tr_idx):.4f}  val macro-F1 {va_f1:.3f}  "
              f"lr {opt.param_groups[0]['lr']:.0e}  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_state, best_f1, best_epoch


runs = []
yt = y[te_idx]
for seed in range(42, 42 + args.seeds):
    print(f"\n=== {args.model}, seed {seed} ===")
    model, state, va_f1, ep = train_once(seed)
    prob = predict_proba(model, te_idx)
    pred = prob.argmax(1)
    vt, vp = video_level(prob, te_idx)
    r = dict(seed=seed, state=state, val_f1=va_f1, epoch=ep, prob=prob, pred=pred,
             acc=accuracy_score(yt, pred), bal=balanced_accuracy_score(yt, pred),
             f1=f1_score(yt, pred, average="macro"),
             vacc=accuracy_score(vt, vp), vf1=f1_score(vt, vp, average="macro"))
    runs.append(r)
    print(f"seed {seed}: test accuracy {r['acc']:.3f}  macro-F1 {r['f1']:.3f}  |  "
          f"per video: accuracy {r['vacc']:.3f}  macro-F1 {r['vf1']:.3f}")

best = max(runs, key=lambda r: r["val_f1"])          # chosen on validation, never on test
ms = lambda k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs]))
pred = best["pred"]

# ---------------- report ----------------
lines = [f"Settings: model={args.model}, features={args.features}, velocity={args.velocity}, augment={args.augment}, "
         f"mirror={args.mirror}, speed={args.speed}, weighting={args.weighting}, exclude_own={args.exclude_own}, "
         f"dropped={sorted(drop) or '-'}, test-pattern={args.test_pattern or '(random 20%)'}",
         f"Classes: {len(actions)}   Test windows: {len(te_idx)}   Test clips: {len(set(groups[te_idx]))}",
         f"Seeds: {args.seeds} (model kept: seed {best['seed']}, best validation macro-F1 {best['val_f1']:.3f}, "
         f"epoch {best['epoch']})", ""]
for key, label in (("acc", "accuracy"), ("bal", "balanced accuracy"), ("f1", "macro-F1"),
                   ("vacc", "per-video accuracy"), ("vf1", "per-video macro-F1")):
    m, s = ms(key)
    lines.append(f"Test {label:20s} {best[key]:.3f}   (mean over seeds {m:.3f} +- {s:.3f})")
for label, mask in (("dataset (unseen people/videos)", is_dataset[te_idx]), ("own recordings", ~is_dataset[te_idx])):
    if mask.any():
        lines.append(f"Accuracy on {label}: {accuracy_score(yt[mask], pred[mask]):.3f}  ({mask.sum()} windows)")
report = classification_report(yt, pred, labels=range(len(actions)), target_names=actions, zero_division=0)
cm = confusion_matrix(yt, pred, labels=range(len(actions)))
mistakes = sorted(((cm[i, j], actions[i], actions[j]) for i in range(len(actions))
                   for j in range(len(actions)) if i != j and cm[i, j] > 0), reverse=True)[:10]
conf = "Most common confusions (true -> predicted):\n" + "\n".join(
    f"  {t:16s} -> {p:16s} {n:5d}  ({n / cm[actions.index(t)].sum():.1%} of {t})" for n, t, p in mistakes)
summary = "\n".join(lines) + "\n\n" + report + "\n" + conf + "\n"
print("\n" + summary)
sfx = f"_{args.tag}" if args.tag else ""
with open(f"results{sfx}.txt", "w") as f:
    f.write(summary)

fig, axes = plt.subplots(1, 2, figsize=(max(16, len(actions) * 1.3), max(7, len(actions) * 0.55)))
ConfusionMatrixDisplay.from_predictions(yt, pred, labels=range(len(actions)), display_labels=actions,
                                        xticks_rotation=60, ax=axes[0], colorbar=False)
axes[0].set_title("Counts")
ConfusionMatrixDisplay.from_predictions(yt, pred, labels=range(len(actions)), display_labels=actions,
                                        xticks_rotation=60, normalize="true", values_format=".2f",
                                        ax=axes[1], colorbar=False)
axes[1].set_title("Row-normalised (recall per class)")
plt.tight_layout()
plt.savefig(f"confusion_matrix{sfx}.png", dpi=130)

torch.save({"state_dict": best["state"], "actions": actions, "features": args.features, "velocity": args.velocity,
            "model": args.model, "hidden": args.hidden, "mean": mean.tolist(), "std": std.tolist()},
           f"har_model{sfx}.pt")

new_file = not os.path.exists(args.ablation_csv)
with open(args.ablation_csv, "a", newline="") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(["tag", "model", "velocity", "augment", "mirror", "speed", "exclude_own", "dropped", "classes",
                    "test_windows", "seeds", "val_macro_f1", "accuracy", "accuracy_sd", "balanced_acc", "macro_f1", "macro_f1_sd",
                    "video_accuracy", "video_macro_f1"])
    w.writerow([args.tag, args.model, args.velocity, args.augment, args.mirror, args.speed, args.exclude_own,
                "|".join(sorted(drop)), len(actions), len(te_idx), args.seeds, f"{ms('val_f1')[0]:.3f}",
                f"{ms('acc')[0]:.3f}", f"{ms('acc')[1]:.3f}", f"{ms('bal')[0]:.3f}", f"{ms('f1')[0]:.3f}",
                f"{ms('f1')[1]:.3f}", f"{ms('vacc')[0]:.3f}", f"{ms('vf1')[0]:.3f}"])
print(f"Saved har_model{sfx}.pt, confusion_matrix{sfx}.png, results{sfx}.txt; row added to {args.ablation_csv}")
