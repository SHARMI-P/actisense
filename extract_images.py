"""Turn folders of posture images (yoga, Bharatanatyam ...) into one file of skeleton features.

python extract_images.py --root ".../yoga-poses-dataset/DATASET/TRAIN" --split train --source yoga --out poses_yoga_train.npz
python extract_images.py --root ".../new_Dataset" --strip " Augmented" --source bharatanatyam --out poses_bharat.npz

Layout: <root>/<pose name>/<image>.jpg|png|jpeg|bmp  (one folder per pose).
Each image -> MediaPipe Pose in still-image mode -> 132 normalised values (same as the live app).
--split: train | test | any (any = train_pose.py splits it).  Images without a detected person are skipped.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True, help="output .npz file")
    p.add_argument("--source", required=True, help="dataset name stored with each sample, e.g. yoga")
    p.add_argument("--split", default="any", choices=["train", "test", "any"])
    p.add_argument("--strip", default="", help="text removed from folder names, e.g. ' Augmented'")
    p.add_argument("--map", default="", help="rename poses: old=new,...")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    return p.parse_args()


def clean(name, strip, rename):
    if strip:
        name = name.replace(strip, "")
    name = name.strip().lower().replace(" ", "_")
    return rename.get(name, name)


_pose = None


def init():
    global _pose
    from common import mp_pose
    _pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)


def work(path):
    import cv2
    from common import landmarks_array, normalize
    img = cv2.imread(path)
    if img is None:
        return path, None
    h, w = img.shape[:2]
    if min(h, w) < 256:                      # tiny images: upscale so the person is found
        s = 256 / min(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    lm = landmarks_array(_pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return path, None if lm is None else normalize(lm)


def main():
    args = parse()
    rename = dict(kv.split("=", 1) for kv in args.map.split(",") if "=" in kv)
    items = []
    for cls in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, cls)
        if not os.path.isdir(d):
            continue
        label = clean(cls, args.strip, rename)
        items += [(os.path.join(d, f), label) for f in sorted(os.listdir(d)) if f.lower().endswith(IMAGE_EXT)]
    print(f"{len(items)} images in {len({l for _, l in items})} poses under {args.root}", flush=True)

    label_of = dict(items)
    X, y, names, found = [], [], [], {}
    with Pool(args.workers, initializer=init) as pool:
        for n, (path, feat) in enumerate(pool.imap_unordered(work, [p for p, _ in items], chunksize=8), 1):
            label = label_of[path]
            found.setdefault(label, [0, 0])[1] += 1
            if feat is not None:
                found[label][0] += 1
                X.append(feat)
                y.append(label)
                names.append(os.path.relpath(path, args.root))
            if n % 500 == 0:
                print(f"  {n}/{len(items)} images", flush=True)

    np.savez_compressed(args.out, X=np.array(X, dtype=np.float32), y=np.array(y), names=np.array(names),
                        source=args.source, split=args.split)
    print(f"Saved {len(X)} samples to {args.out}")
    for label, (ok, total) in sorted(found.items()):
        print(f"  {label:20s} person found in {ok}/{total} images")


if __name__ == "__main__":
    main()
