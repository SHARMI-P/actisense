"""Convert a folder of labelled clips (e.g. the KTH dataset) into training windows, no webcam needed.

Expected layout (one sub-folder per class). A clip is either a video file or a folder of image frames:
    videos/
      walking/    person01_walking_d1_uncomp.avi ...
      handwaving/ ...

Usage:
    python extract_dataset.py --root videos
    python extract_dataset.py --root videos --map handwaving=waving --skip jogging,running

Each clip is cut into overlapping 30-frame windows of pose features, saved as
data/<class>/<clip-name>_<n>.npy, so they can be mixed with webcam recordings.
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import mp_pose, landmarks_array, normalize, find_sequences, iter_frames, SEQ_LEN


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder with one sub-folder per class")
    p.add_argument("--out", default="data")
    p.add_argument("--stride", type=int, default=10, help="frames between window starts")
    p.add_argument("--min-side", type=int, default=480,
                   help="upscale small videos (KTH is 160x120) so the pose model sees the person")
    p.add_argument("--map", default="", help="rename classes, e.g. handwaving=waving,handclapping=clapping")
    p.add_argument("--skip", default="", help="classes to ignore, comma separated")
    return p.parse_args()


def clip_windows(path, stride, min_side):
    buf, windows, frames, detected = [], [], 0, 0
    # a fresh Pose object per clip so tracking does not leak between clips
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for frame in iter_frames(path, min_side):
            frames += 1
            lm = landmarks_array(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if lm is None:
                continue                  # person not found in this frame: skip it
            detected += 1
            buf.append(normalize(lm))
            if len(buf) == SEQ_LEN:
                windows.append(np.stack(buf))
                buf = buf[stride:]
    return windows, frames, detected


def main():
    args = parse()
    rename = dict(kv.split("=", 1) for kv in args.map.split(",") if "=" in kv)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    classes = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    if not classes:
        raise SystemExit(f"No class folders found in {args.root}")

    # list of all clips first, so we can show progress and an estimated finish time
    jobs = []
    for cls in classes:
        if cls in skip:
            continue
        label = rename.get(cls, cls)
        jobs += [(label, clip) for clip in find_sequences(os.path.join(args.root, cls))]

    # resume support: clips already finished in an earlier (interrupted) run are skipped
    os.makedirs(args.out, exist_ok=True)
    done_file = os.path.join(args.out, ".extracted.txt")
    done = set(open(done_file, encoding="utf-8").read().splitlines()) if os.path.exists(done_file) else set()
    todo = [(l, c) for l, c in jobs if os.path.abspath(c) not in done]
    print(f"{len(jobs)} clips found, {len(jobs) - len(todo)} already done, {len(todo)} to process.\n")

    summary, t0 = {}, time.time()
    for n, (label, clip) in enumerate(todo, 1):
        out_dir = os.path.join(args.out, label)
        os.makedirs(out_dir, exist_ok=True)
        wins, frames, detected = clip_windows(clip, args.stride, args.min_side)
        stem = os.path.splitext(os.path.basename(clip.rstrip("/\\")))[0]
        for i, w in enumerate(wins):
            np.save(os.path.join(out_dir, f"{stem}_{i:03d}.npy"), w.astype(np.float32))
        with open(done_file, "a", encoding="utf-8") as f:
            f.write(os.path.abspath(clip) + "\n")
        summary[label] = summary.get(label, 0) + len(wins)
        rate = detected / frames if frames else 0
        left = (time.time() - t0) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] [{label}] {stem}: {len(wins)} windows, person found in {rate:.0%} "
              f"of frames | about {left / 60:.0f} min left", flush=True)

    print("\nWindows added per class in this run:")
    for k, v in sorted(summary.items()):
        print(f"  {k:15s} {v}")
    print("\nTotal windows now in", args.out)
    for d in sorted(os.listdir(args.out)):
        if os.path.isdir(os.path.join(args.out, d)):
            print(f"  {d:15s} {len([f for f in os.listdir(os.path.join(args.out, d)) if f.endswith('.npy')])}")


if __name__ == "__main__":
    main()