"""Measure the fall detector on the UR Fall Detection dataset (or any falls / non-falls clips).

Layout (use the camera-0 RGB zips; cam0 is the side view, cam1 is ceiling-mounted):
    urfall/
      falls/  fall-01-cam0-rgb/ (PNG frames)  fall-02-cam0-rgb/ ...
      adls/   adl-01-cam0-rgb/  adl-02-cam0-rgb/ ...

python evaluate_falls.py --root urfall

The pose of every frame is computed once and cached in urfall_poses.npz, then the rule is tried
with several (drop, tilt) settings. Results go to fall_eval.csv for the report.
"""
import argparse
import csv
import os

import cv2
import numpy as np

from common import mp_pose, landmarks_array, find_sequences, iter_frames, FallDetector

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--cache", default="urfall_poses.npz")
p.add_argument("--out", default="fall_eval.csv")
args = p.parse_args()


def extract(path):
    """Per-frame landmarks (None where no person) plus frame size."""
    seq, size = [], (1, 1)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for frame in iter_frames(path):
            size = frame.shape[1], frame.shape[0]
            seq.append(landmarks_array(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    return seq, size


# ---- 1. pose extraction (slow, done once and cached) ----
if os.path.exists(args.cache):
    cache = np.load(args.cache, allow_pickle=True)["data"].item()
    print(f"Loaded cached poses for {len(cache)} clips from {args.cache}")
else:
    cache = {}
    for label in ("falls", "adls"):
        clips = find_sequences(os.path.join(args.root, label))
        if not clips:
            raise SystemExit(f"No clips found in {os.path.join(args.root, label)}")
        for clip in clips:
            seq, size = extract(clip)
            name = os.path.basename(clip.rstrip("/\\"))
            found = sum(x is not None for x in seq) / max(len(seq), 1)
            print(f"[{label}] {name}: {len(seq)} frames, person found in {found:.0%}")
            cache[name] = {"label": label, "seq": seq, "size": size}
    np.savez_compressed(args.cache, data=np.array(cache, dtype=object))


# ---- 2. evaluate the rule at several settings (fast) ----
def detects_fall(entry, drop, tilt):
    det = FallDetector(drop=drop, tilt=tilt)
    w, h = entry["size"]
    return any(det.update(lm, w, h) for lm in entry["seq"] if lm is not None)


rows = []
for drop in (0.10, 0.15, 0.20, 0.25):
    for tilt in (45, 60, 75):
        tp = fn = fp = tn = 0
        for e in cache.values():
            hit = detects_fall(e, drop, tilt)
            if e["label"] == "falls":
                tp, fn = tp + hit, fn + (not hit)
            else:
                fp, tn = fp + hit, tn + (not hit)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        rows.append([drop, tilt, tp, fn, fp, tn, round(sens, 3), round(spec, 3), round(acc, 3)])

header = ["drop", "tilt", "TP", "FN", "FP", "TN", "sensitivity", "specificity", "accuracy"]
with open(args.out, "w", newline="") as f:
    csv.writer(f).writerows([header] + rows)

print("\n" + "  ".join(f"{h:>11s}" for h in header))
for r in rows:
    print("  ".join(f"{v:>11}" for v in r))
best = max(rows, key=lambda r: (r[6] + r[7], r[8]))
print(f"\nBest balance: drop={best[0]}, tilt={best[1]} -> sensitivity {best[6]}, specificity {best[7]}")
print("Sensitivity = falls caught; specificity = normal activities NOT flagged as falls.")
print(f"Full table saved to {args.out}. Use the best values in FallDetector() inside run_app.py.")
