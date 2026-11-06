"""Turn the Penn Action dataset (sports / exercise videos as frame folders) into 30-frame pose windows.

python extract_penn.py --root /kaggle/input/.../Penn_Action --out data_penn --workers 4

Penn_Action/frames/0001/000001.jpg ...  +  Penn_Action/labels/0001.mat (action name, official train/test flag)
Windows are saved as data_penn/<activity>/pennTrain_0001_000.npy or pennTest_0001_000.npy, so train.py
can keep Penn's official test videos for testing (--test-pattern "...|pennTest_").
Re-running skips videos that are already done.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
from scipy.io import loadmat

# Penn Action names -> ActiSense activity names. "squat" joins the existing "squatting" class.
DEFAULT_ACTIONS = ("jumping_jacks=jumping_jacks,pushup=pushup,situp=situp,pullup=pullup,jump_rope=jump_rope,"
                   "squat=squatting,tennis_serve=tennis_serve,tennis_forehand=tennis_forehand,"
                   "golf_swing=golf_swing,baseball_swing=baseball_swing,bowl=bowling")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="the Penn_Action folder (holds frames/ and labels/)")
    p.add_argument("--out", default="data_penn")
    p.add_argument("--actions", default=DEFAULT_ACTIONS, help="penn_name=actisense_name,... (others are skipped)")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--min-side", type=int, default=480)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return p.parse_args()


def read_label(path):
    m = loadmat(path)
    action = str(np.atleast_1d(m["action"]).ravel()[0]).strip()
    train = int(np.atleast_1d(m["train"]).ravel()[0]) == 1
    return action, train


def work(job):
    vid, frames_dir, label, split, out, stride, min_side = job
    import cv2
    from common import mp_pose, landmarks_array, normalize, iter_frames, SEQ_LEN
    buf, n_win, frames, found = [], 0, 0, 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for frame in iter_frames(frames_dir, min_side):
            frames += 1
            lm = landmarks_array(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if lm is None:
                continue
            found += 1
            buf.append(normalize(lm))
            if len(buf) == SEQ_LEN:
                np.save(os.path.join(out, label, f"{split}_{vid}_{n_win:03d}.npy"), np.stack(buf).astype(np.float32))
                n_win += 1
                buf = buf[stride:]
    return vid, label, n_win, found / max(frames, 1)


def main():
    args = parse()
    mapping = dict(kv.split("=", 1) for kv in args.actions.split(",") if "=" in kv)
    labels_dir, frames_root = os.path.join(args.root, "labels"), os.path.join(args.root, "frames")
    done_file = os.path.join(args.out, ".extracted_penn.txt")
    os.makedirs(args.out, exist_ok=True)
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()

    jobs, skipped = [], {}
    for f in sorted(os.listdir(labels_dir)):
        if not f.endswith(".mat"):
            continue
        vid = os.path.splitext(f)[0]
        action, train = read_label(os.path.join(labels_dir, f))
        if action not in mapping:
            skipped[action] = skipped.get(action, 0) + 1
            continue
        if vid in done:
            continue
        label = mapping[action]
        os.makedirs(os.path.join(args.out, label), exist_ok=True)
        jobs.append((vid, os.path.join(frames_root, vid), label, "pennTrain" if train else "pennTest",
                     args.out, args.stride, args.min_side))
    print(f"{len(jobs)} videos to process ({len(done)} already done); skipped actions: {skipped}", flush=True)

    totals = {}
    with Pool(args.workers) as pool, open(done_file, "a") as log:
        for n, (vid, label, n_win, rate) in enumerate(pool.imap_unordered(work, jobs), 1):
            log.write(vid + "\n")
            log.flush()
            totals[label] = totals.get(label, 0) + n_win
            if n % 50 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] last: {vid} {label} {n_win} windows, person found {rate:.0%}", flush=True)
    print("\nWindows added per activity:", dict(sorted(totals.items())))


if __name__ == "__main__":
    main()
