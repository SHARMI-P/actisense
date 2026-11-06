"""Train the posture classifier (yoga, Bharatanatyam ...) on single skeleton frames.

python train_pose.py --poses poses_yoga_train.npz poses_yoga_test.npz poses_bharat.npz \
                     --negatives data data_penn --test-pattern "person(02|03|05|06|07|08|09|10|22)_|pennTest_"

--poses      .npz files from extract_images.py (their split: train / test / any = random 20% test)
--negatives  activity window folders; one frame per window becomes the class "none" (= no held posture),
             so the classifier can say "not a posture" instead of forcing one.
Saves pose_model.pt, results_pose.txt and confusion_matrix_pose.png.
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
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.model_selection import train_test_split

from common import PoseMLP, pose_vector, mirror_xyz

ap = argparse.ArgumentParser()
ap.add_argument("--poses", nargs="+", required=True)
ap.add_argument("--negatives", nargs="*", default=[])
ap.add_argument("--neg-per-class", type=int, default=250, help="windows taken per activity for 'none'")
ap.add_argument("--test-pattern", default="", help="negative files matching this regex go to test")
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--patience", type=int, default=20)
ap.add_argument("--tag", default="")
args = ap.parse_args()
rng = np.random.default_rng(42)
torch.manual_seed(42)

# ---------------- load posture images ----------------
xyz, labels, split, source = [], [], [], []
for f in args.poses:
    d = np.load(f, allow_pickle=False)
    n = len(d["y"])
    xyz.append(d["X"][:, :99].reshape(n, 33, 3))
    labels += [str(v) for v in d["y"]]
    split += [str(d["split"])] * n
    source += [str(d["source"])] * n
    print(f"{f}: {n} samples, source={d['source']}, split={d['split']}")

# ---------------- 'none' = one frame from activity windows ----------------
for root in args.negatives:
    for cls in sorted(os.listdir(root)):
        files = sorted(glob.glob(os.path.join(root, cls, "*.npy")))
        if not files:
            continue
        pick = rng.choice(len(files), size=min(args.neg_per_class, len(files)), replace=False)
        for i in pick:
            w = np.load(files[i])
            xyz.append(w[len(w) // 2, :99].reshape(1, 33, 3))
            labels.append("none")
            name = os.path.basename(files[i])
            is_test = bool(args.test_pattern and re.search(args.test_pattern, name))
            split.append("test" if is_test else ("train" if re.match(r"^(person\d+_|penn)", name) else "any"))
            source.append("activities")
    print(f"negatives from {root}: total 'none' so far {labels.count('none')}")

XYZ = np.concatenate(xyz).astype(np.float32)
labels, split, source = np.array(labels), np.array(split), np.array(source)
classes = sorted(str(c) for c in set(labels.tolist()))
y = np.array([classes.index(l) for l in labels], dtype=np.int64)
print(f"\nClasses ({len(classes)}): {classes}\nSamples: {len(y)}")

# ---------------- split: given train/test, 'any' -> stratified 20% test; then 15% validation ----------------
idx = np.arange(len(y))
te = list(idx[split == "test"])
tr = list(idx[split == "train"])
any_idx = idx[split == "any"]
if len(any_idx):
    a_tr, a_te = train_test_split(any_idx, test_size=0.2, stratify=y[any_idx], random_state=42)
    tr += list(a_tr)
    te += list(a_te)
tr, va = train_test_split(np.array(tr), test_size=0.15, stratify=y[tr], random_state=7)
te = np.array(te)
print(f"Split: train {len(tr)} / validation {len(va)} / test {len(te)}")


def features(batch_xyz):
    return np.stack([pose_vector(p) for p in batch_xyz])


def augment(batch_xyz):
    out = batch_xyz.copy()
    flip = rng.random(len(out)) < 0.5
    out[flip] = mirror_xyz(out[flip])
    th = rng.uniform(-np.radians(20), np.radians(20), len(out))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    x, z = out[..., 0].copy(), out[..., 2].copy()
    out[..., 0], out[..., 2] = c * x + s * z, -s * x + c * z
    out *= rng.uniform(0.9, 1.1, (len(out), 1, 1))
    out += rng.normal(0, 0.02, out.shape).astype(np.float32)
    return out


F_all = features(XYZ)
mean, std = F_all[tr].mean(0), F_all[tr].std(0) + 1e-6
norm = lambda F: torch.from_numpy(((F - mean) / std).astype(np.float32))

counts = np.bincount(y[tr], minlength=len(classes)).astype(np.float32)
weights = np.sqrt(counts.sum() / (len(classes) * np.maximum(counts, 1)))
weights /= weights.mean()
model = PoseMLP(F_all.shape[1], len(classes))
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))


def predict(ids):
    model.eval()
    with torch.no_grad():
        return model(norm(F_all[ids])).argmax(1).numpy()


best, best_state, best_ep, bad = -1.0, None, 0, 0
for ep in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    order = rng.permutation(tr)
    for i in range(0, len(order), 128):
        b = order[i:i + 128]
        xb = norm(features(augment(XYZ[b])))
        opt.zero_grad()
        loss = loss_fn(model(xb), torch.from_numpy(y[b]))
        loss.backward()
        opt.step()
    f1 = f1_score(y[va], predict(va), average="macro")
    sched.step(f1)
    if f1 > best:
        best, best_ep, bad = f1, ep, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    if ep % 5 == 0 or ep == 1:
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val macro-F1 {f1:.3f}  ({time.time() - t0:.1f}s)", flush=True)
    if bad >= args.patience:
        print(f"No improvement for {args.patience} epochs - stopping.")
        break

model.load_state_dict(best_state)
pred, yt = predict(te), y[te]
lines = [f"Best epoch (validation): {best_ep}", f"Test samples: {len(te)}",
         f"Test accuracy:          {accuracy_score(yt, pred):.3f}",
         f"Test balanced accuracy: {balanced_accuracy_score(yt, pred):.3f}",
         f"Test macro-F1:          {f1_score(yt, pred, average='macro'):.3f}"]
for s in sorted(set(source[te])):
    m = source[te] == s
    lines.append(f"Accuracy on {s}: {accuracy_score(yt[m], pred[m]):.3f}  ({m.sum()} samples)")
lines.append("Note: samples with split 'any' were split at random; augmented copies of one photo can fall on "
             "both sides, so those scores can be optimistic.")
cm = confusion_matrix(yt, pred, labels=range(len(classes)))
mist = sorted(((cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes))
               if i != j and cm[i, j]), reverse=True)[:8]
report = classification_report(yt, pred, labels=range(len(classes)), target_names=classes, zero_division=0)
summary = "\n".join(lines) + "\n\n" + report + "\nMost common confusions (true -> predicted):\n" + \
    "\n".join(f"  {t:16s} -> {p:16s} {n}" for n, t, p in mist) + "\n"
print("\n" + summary)
sfx = f"_{args.tag}" if args.tag else ""
open(f"results_pose{sfx}.txt", "w").write(summary)

fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.7), max(7, len(classes) * 0.6)))
ConfusionMatrixDisplay.from_predictions(yt, pred, labels=range(len(classes)), display_labels=classes,
                                        normalize="true", values_format=".2f", xticks_rotation=60, ax=ax,
                                        colorbar=False)
ax.set_title("Posture classifier – recall per class")
plt.tight_layout()
plt.savefig(f"confusion_matrix_pose{sfx}.png", dpi=140)
torch.save({"state_dict": best_state, "classes": classes, "mean": mean.tolist(), "std": std.tolist(),
            "n_features": int(F_all.shape[1])}, f"pose_model{sfx}.pt")
print(f"Saved pose_model{sfx}.pt, results_pose{sfx}.txt, confusion_matrix_pose{sfx}.png")
