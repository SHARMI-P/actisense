"""Real-time recognition + analysis: activity label, squat rep counter, fall alert, CSV session log.

python run_app.py                 (webcam)
python run_app.py --source x.mp4  (video file)
Press q to quit. The log is written to logs/session.csv (one row per second).
"""
import argparse
import csv
import os
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch

from common import (HARModel, select_features, SEQ_LEN, mp_pose, mp_draw, landmarks_array,
                    normalize, joint_angle, FallDetector, L_HIP, L_KNEE, L_ANKLE)

p = argparse.ArgumentParser()
p.add_argument("--source", default="0")
p.add_argument("--model", default="har_model.pt")
p.add_argument("--log", default="logs/session.csv")
p.add_argument("--threshold", type=float, default=0.6, help="min probability to accept a prediction")
args = p.parse_args()

ckpt = torch.load(args.model, map_location="cpu")
actions = ckpt["actions"]
mean = np.array(ckpt["mean"], dtype=np.float32)
std = np.array(ckpt["std"], dtype=np.float32)
features = ckpt.get("features", "all")      # older models were trained on all 132 values
model = HARModel(len(mean), len(actions))
model.load_state_dict(ckpt["state_dict"])
model.eval()

cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
if not cap.isOpened():
    raise SystemExit(f"Cannot open source: {args.source}")

os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
log_f = open(args.log, "w", newline="")
writer = csv.writer(log_f)
writer.writerow(["timestamp", "activity", "confidence", "squat_reps", "fall_alert"])

window = deque(maxlen=SEQ_LEN)      # pose features for the classifier
recent = deque(maxlen=8)            # recent predictions for smoothing
fall_det = FallDetector(drop=0.15, tilt=45)           # same rule that evaluate_falls.py measures on UR Fall
activity, conf = "unknown", 0.0
reps, stage = 0, None
fall, fall_time, last_log = False, 0.0, 0.0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm = landmarks_array(res)

        if lm is not None:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            window.append(normalize(lm))
            px = lm[:, :2] * np.array([w, h], dtype=np.float32)   # pixel coords for angles

            # --- squat repetition counter (knee angle state machine) ---
            knee = joint_angle(px[L_HIP], px[L_KNEE], px[L_ANKLE])
            if knee < 100:
                stage = "down"
            elif knee > 160 and stage == "down":
                stage = "up"
                if activity == "squatting":
                    reps += 1

            # --- fall heuristic: fast hip drop + torso close to horizontal ---
            if fall_det.update(lm, w, h):
                fall, fall_time = True, time.time()

        if fall and time.time() - fall_time > 5:
            fall = False

        # --- activity classification on the last SEQ_LEN frames ---
        if len(window) == SEQ_LEN:
            x = (select_features(np.stack(window), features) - mean) / std
            with torch.no_grad():
                prob = torch.softmax(model(torch.from_numpy(x[None]).float()), 1)[0].numpy()
            if prob.max() >= args.threshold:
                recent.append(int(prob.argmax()))
            if recent:
                idx = Counter(recent).most_common(1)[0][0]
                activity, conf = actions[idx], float(prob[idx])

        now = time.time()
        if now - last_log >= 1.0:
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), activity, f"{conf:.2f}", reps, int(fall)])
            log_f.flush()
            last_log = now

        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.putText(frame, f"{activity} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Squat reps: {reps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if fall:
            cv2.putText(frame, "FALL DETECTED", (w - 300, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("ActiSense", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
log_f.close()
print(f"Session log saved to {args.log}")
