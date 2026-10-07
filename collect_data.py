"""Record labelled pose sequences from a webcam or a video file.

Webcam:  python collect_data.py --action walking --sequences 40
Video:   python collect_data.py --action walking --source path/to/video.avi
Press q to stop early.
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import mp_pose, mp_draw, landmarks_array, normalize, SEQ_LEN


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--action", required=True, help="class label, e.g. walking")
    p.add_argument("--source", default="0", help="webcam index (0) or a video file path")
    p.add_argument("--sequences", type=int, default=40, help="windows to record (webcam only)")
    p.add_argument("--stride", type=int, default=15, help="window step for video files")
    return p.parse_args()


def save(seq, out_dir):
    idx = len([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    np.save(os.path.join(out_dir, f"{idx:04d}.npy"), np.stack(seq))


def main():
    args = parse()
    out_dir = os.path.join("data", args.action)
    os.makedirs(out_dir, exist_ok=True)

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    saved, buf, stop = 0, [], False
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while not stop and (not is_cam or saved < args.sequences):
            if is_cam and not buf:                      # 2-second pause between recordings
                t0 = time.time()
                while time.time() - t0 < 2:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    cv2.putText(frame, f"Get ready: {args.action} ({saved + 1}/{args.sequences})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("collect", frame)
                    cv2.waitKey(1)

            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = landmarks_array(res)
            if lm is not None:
                buf.append(normalize(lm))
                mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if len(buf) == SEQ_LEN:
                save(buf, out_dir)
                saved += 1
                buf = [] if is_cam else buf[args.stride:]

            cv2.putText(frame, f"{args.action}: saved {saved}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("collect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop = True

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} sequences to {out_dir}")


if __name__ == "__main__":
    main()
