"""Train the LSTM classifier on data/<action>/*.npy and save har_model.pt, confusion_matrix.png, results.txt.

python train.py
python train.py --test-pattern "person(02|03|05|06|07|08|09|10|22)_"     # KTH official test people

Data split (three parts, never mixed):
  test       - KTH people matching --test-pattern + 20% of your own (webcam) recordings
  validation - 15% of the remaining clips; used ONLY to pick the best epoch / stop early
  train      - everything else
Windows cut from the same video always stay in the same part, so nothing leaks between them.
Class imbalance (e.g. 5000 waving vs 300 sitting windows) is handled with a class-weighted loss.
"""
import argparse
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
                             ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from common import HARModel, select_features

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--patience", type=int, default=10, help="stop after this many epochs without improvement")
ap.add_argument("--weighting", choices=["inverse", "sqrt", "none"], default="sqrt",
                help="class weights: inverse = strong (high recall, low precision for rare classes), "
                     "sqrt = softer (default), none = no weighting")
ap.add_argument("--features", choices=["all", "xyz"], default="all",
                help="xyz = drop MediaPipe visibility scores (removes a 'which camera' shortcut)")
ap.add_argument("--augment", action="store_true",
                help="random rotation / scale / jitter of skeletons during training (camera robustness)")
ap.add_argument("--tag", default="", help="suffix for output files, e.g. v2 -> results_v2.txt")
ap.add_argument("--test-pattern", default="",
                help="regex on file names; matching dataset files form the test set (held-out people)")
args = ap.parse_args()

BATCH, LR = 64, 1e-3
torch.manual_seed(42)
np.random.seed(42)
SUBJECT_FILE = re.compile(r"^person\d+_")        # files that come from KTH (have a person id)

# ---------------- load ----------------
actions = sorted(d for d in os.listdir(args.data) if glob.glob(os.path.join(args.data, d, "*.npy")))
if len(actions) < 2:
    raise SystemExit(f"Need recorded data for at least 2 actions in {args.data}/")

X, y, names, groups = [], [], [], []
for i, a in enumerate(actions):
    for f in sorted(glob.glob(os.path.join(args.data, a, "*.npy"))):
        X.append(np.load(f))
        y.append(i)
        base = os.path.basename(f)
        names.append(base)
        groups.append(a + "/" + re.sub(r"_\d+\.npy$", "", base))   # one group per source clip
X = select_features(np.stack(X).astype(np.float32), args.features)
NF = X.shape[-1]
y = np.array(y, dtype=np.int64)          # int64 explicitly: Windows numpy defaults to int32, PyTorch needs int64
names, groups = np.array(names), np.array(groups)
is_dataset = np.array([bool(SUBJECT_FILE.match(n)) for n in names])
print(f"Classes: {actions}\nWindows: {len(X)}  (dataset {is_dataset.sum()}, own recordings {(~is_dataset).sum()})")


def group_split(idx, test_size, seed=42):
    """Split indices so that all windows of one clip land on the same side."""
    if len(idx) == 0:
        return idx, idx
    if len(set(groups[idx])) < 2:
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
print(f"Split: train {len(tr_idx)} / validation {len(va_idx)} / test {len(te_idx)} windows")
missing = [actions[c] for c in range(len(actions)) if c not in set(y[te_idx])]
if missing:
    print(f"WARNING: no test windows for {missing} - record more clips of these activities.")

# ---------------- normalise with training statistics only ----------------
mean = X[tr_idx].reshape(-1, NF).mean(0)
std = X[tr_idx].reshape(-1, NF).std(0) + 1e-6
Xn = (X - mean) / std
t = lambda idx: torch.from_numpy(Xn[idx])
mean_t, std_t = torch.from_numpy(mean), torch.from_numpy(std)

# the loader gives RAW skeletons, so augmentation happens before normalisation
train_dl = DataLoader(TensorDataset(torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])),
                      batch_size=BATCH, shuffle=True)


def augment(xb):
    """Random camera-like changes to a batch of skeleton sequences (B, 30, NF), xyz = first 99 values.
    The same change is applied to all 30 frames of a sequence."""
    xb = xb.clone()
    B = xb.shape[0]
    xyz = xb[..., :99].reshape(B, xb.shape[1], 33, 3)
    theta = (torch.rand(B) * 2 - 1) * np.radians(20)            # turn left/right up to 20 degrees
    c, s = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
    x, z = xyz[..., 0].clone(), xyz[..., 2].clone()
    xyz[..., 0], xyz[..., 2] = c * x + s * z, -s * x + c * z
    xyz *= (0.85 + 0.3 * torch.rand(B))[:, None, None, None]    # body size / distance
    xyz += 0.02 * torch.randn_like(xyz)                          # keypoint noise
    xb[..., :99] = xyz.reshape(B, xb.shape[1], 99)
    return xb

# class weights: rare activities count more, so the model cannot ignore them
counts = np.bincount(y[tr_idx], minlength=len(actions)).astype(np.float32)
weights = counts.sum() / (len(actions) * np.maximum(counts, 1))
if args.weighting == "sqrt":
    weights = np.sqrt(weights)
elif args.weighting == "none":
    weights = np.ones_like(weights)
weights = weights / weights.mean()
print("Train windows per class:", dict(zip(actions, counts.astype(int).tolist())))
print(f"Class weights ({args.weighting}):", {a: round(float(w), 2) for a, w in zip(actions, weights)})

model = HARModel(NF, len(actions))
opt = torch.optim.Adam(model.parameters(), lr=LR)
# halve the learning rate when validation stops improving -> calmer, more stable training
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))


def predict(idx):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(t(idx[i:i + 1024])).argmax(1) for i in range(0, len(idx), 1024)]).numpy()


# ---------------- train, choosing the best epoch on VALIDATION (never on test) ----------------
best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    total = 0.0
    for xb, yb in train_dl:
        if args.augment:
            xb = augment(xb)
        xb = (xb - mean_t) / std_t
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # prevents the loss spikes LSTMs get
        opt.step()
        total += loss.item() * len(xb)
    va_f1 = f1_score(y[va_idx], predict(va_idx), average="macro") if len(va_idx) else 0.0
    sched.step(va_f1)
    if va_f1 > best_f1:
        best_f1, best_epoch, bad = va_f1, epoch, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    print(f"epoch {epoch:3d}  loss {total / len(tr_idx):.4f}  val macro-F1 {va_f1:.3f}  "
          f"lr {opt.param_groups[0]['lr']:.0e}  ({time.time() - t0:.0f}s)",
          flush=True)
    if bad >= args.patience:
        print(f"No improvement for {args.patience} epochs - stopping early.")
        break

# ---------------- final, one-time evaluation on TEST ----------------
model.load_state_dict(best_state)
pred = predict(te_idx)
yt = y[te_idx]
lines = [
    f"Best epoch (chosen on validation): {best_epoch}",
    f"Test windows: {len(te_idx)}",
    f"Test accuracy:          {accuracy_score(yt, pred):.3f}",
    f"Test balanced accuracy: {balanced_accuracy_score(yt, pred):.3f}",
    f"Test macro-F1:          {f1_score(yt, pred, average='macro'):.3f}",
]
for label, mask in (("dataset (unseen people)", is_dataset[te_idx]), ("own recordings", ~is_dataset[te_idx])):
    if mask.any():
        lines.append(f"Accuracy on {label}: {accuracy_score(yt[mask], pred[mask]):.3f}  ({mask.sum()} windows)")
report = classification_report(yt, pred, labels=range(len(actions)), target_names=actions, zero_division=0)
# the most frequent mistakes, in plain words
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yt, pred, labels=range(len(actions)))
mistakes = sorted(((cm[i, j], actions[i], actions[j]) for i in range(len(actions))
                   for j in range(len(actions)) if i != j and cm[i, j] > 0), reverse=True)[:8]
conf = "Most common confusions (true -> predicted):\n" + "\n".join(
    f"  {t:10s} -> {p:10s} {n:5d}  ({n / cm[actions.index(t)].sum():.1%} of {t})" for n, t, p in mistakes)
summary = (f"Settings: weighting={args.weighting}, features={args.features}, augment={args.augment}, test-pattern={args.test_pattern or '(random 20%)'}\n"
           + "\n".join(lines) + "\n\n" + report + "\n" + conf + "\n")
print("\n" + summary)
sfx = f"_{args.tag}" if args.tag else ""
with open(f"results{sfx}.txt", "w") as f:
    f.write(summary)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ConfusionMatrixDisplay.from_predictions(yt, pred, labels=range(len(actions)), display_labels=actions,
                                        xticks_rotation=45, ax=axes[0], colorbar=False)
axes[0].set_title("Counts")
ConfusionMatrixDisplay.from_predictions(yt, pred, labels=range(len(actions)), display_labels=actions,
                                        xticks_rotation=45, normalize="true", values_format=".2f",
                                        ax=axes[1], colorbar=False)
axes[1].set_title("Row-normalised (recall per class)")
plt.tight_layout()
plt.savefig(f"confusion_matrix{sfx}.png", dpi=150)

torch.save({"state_dict": best_state, "actions": actions, "features": args.features,
            "mean": mean.tolist(), "std": std.tolist()}, f"har_model{sfx}.pt")
print(f"Saved har_model{sfx}.pt, confusion_matrix{sfx}.png and results{sfx}.txt")