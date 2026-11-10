"""Real-time recognition + analysis: activity label, held posture, squat rep counter, fall alert, CSV log.

python run_app.py                 (webcam)
python run_app.py --source x.mp4  (video file)
If pose_model.pt is present, a held posture (yoga, Bharatanatyam) is shown when the person keeps still.
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

from common import (HARModel, PoseMLP, select_features, pose_vector, motion_energy, SEQ_LEN, mp_pose,
                    mp_draw, landmarks_array, normalize, joint_angle, FallDetector, L_HIP, L_KNEE, L_ANKLE)

p = argparse.ArgumentParser()
p.add_argument("--source", default="0")
p.add_argument("--model", default="har_model.pt")
p.add_argument("--log", default="logs/session.csv")
p.add_argument("--threshold", type=float, default=0.6, help="min probability to accept a prediction")
p.add_argument("--pose-model", default="pose_model.pt", help="posture classifier (skipped if the file is missing)")
p.add_argument("--pose-threshold", type=float, default=0.6, help="min probability to show a posture")
p.add_argument("--still", type=float, default=0.03, help="max movement (torso lengths/frame) to count as holding still")
p.add_argument("--smooth", type=int, default=15, help="how many recent predictions vote on the label")
p.add_argument("--legs", type=float, default=0.5, help="min visibility of knees+ankles to trust leg-based labels")
p.add_argument("--debug", action="store_true", help="show movement level and top-3 guesses of both models")
args = p.parse_args()

ckpt = torch.load(args.model, map_location="cpu")
actions = ckpt["actions"]
mean = np.array(ckpt["mean"], dtype=np.float32)
std = np.array(ckpt["std"], dtype=np.float32)
features = ckpt.get("features", "all")      # older models were trained on all 132 values
model = HARModel(len(mean), len(actions))
model.load_state_dict(ckpt["state_dict"])
model.eval()

pose_model = None
if os.path.exists(args.pose_model):
    pck = torch.load(args.pose_model, map_location="cpu")
    pose_classes = pck["classes"]
    pose_mean = np.array(pck["mean"], dtype=np.float32)
    pose_std = np.array(pck["std"], dtype=np.float32)
    pose_model = PoseMLP(pck["n_features"], len(pose_classes))
    pose_model.load_state_dict(pck["state_dict"])
    pose_model.eval()
    print(f"Posture model loaded: {len(pose_classes)} classes")
pose_recent = deque(maxlen=10)       # recent posture predictions for smoothing
posture, pose_conf = "", 0.0

cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
if not cap.isOpened():
    raise SystemExit(f"Cannot open source: {args.source}")

os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
log_f = open(args.log, "w", newline="")
writer = csv.writer(log_f)
writer.writerow(["timestamp", "activity", "confidence", "squat_reps", "fall_alert", "pose"])

window = deque(maxlen=SEQ_LEN)      # pose features for the classifier
recent = deque(maxlen=args.smooth)  # recent predictions for smoothing
STATIC = [a for a in ("standing", "sitting", "lying") if a in actions]   # the only activities allowed while still
motion, still, act_top, pose_top = 0.0, False, [], []
legs_visible = True                 # knees + ankles in view? sitting/standing/postures depend on them
LEG_POINTS = [25, 26, 27, 28]        # MediaPipe: left/right knee, left/right ankle
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
            legs = lm[LEG_POINTS]
            in_frame = (legs[:, 0] >= 0) & (legs[:, 0] <= 1) & (legs[:, 1] >= 0) & (legs[:, 1] <= 1)
            legs_visible = int((in_frame & (legs[:, 3] >= args.legs)).sum()) >= 3   # 3 of 4 knee/ankle points
            px = lm[:, :2] * np.array([w, h], dtype=np.float32)   # pixel coords for angles

            # --- squat repetition counter (knee angle state machine) ---
            knee = joint_angle(px[L_HIP], px[L_KNEE], px[L_ANKLE])
            if knee < 100:
                stage = "down"
            elif knee > 160 and stage == "down":
                stage = "up"
                if activity == "squatting":
                    reps += 1

            # --- movement level over the last 0.5 s: still = holding a posture, else moving ---
            if len(window) >= 15:
                motion = motion_energy(list(window)[-15:])
                still = motion < args.still

            # --- held posture: one-frame classifier, used only while the person keeps still ---
            if pose_model is not None:
                fv = (pose_vector(window[-1]) - pose_mean) / pose_std
                with torch.no_grad():
                    pp = torch.softmax(pose_model(torch.from_numpy(fv[None]).float()), 1)[0].numpy()
                pose_top = [(pose_classes[i], float(pp[i])) for i in np.argsort(pp)[::-1][:3]]
                pose_recent.append(int(pp.argmax()) if pp.max() >= args.pose_threshold else -1)
                best, votes = Counter(pose_recent).most_common(1)[0]
                name = pose_classes[best] if best >= 0 else "none"
                if still and legs_visible and name != "none" and votes >= pose_recent.maxlen // 2:
                    posture, pose_conf = name, float(pp[best])
                else:
                    posture, pose_conf = "", 0.0

            # --- fall heuristic: fast hip drop + torso close to horizontal ---
            if fall_det.update(lm, w, h):
                fall, fall_time = True, time.time()

        if fall and time.time() - fall_time > 5:
            fall = False

        # --- activity classification on the last SEQ_LEN frames (motion-aware) ---
        if len(window) == SEQ_LEN:
            x = (select_features(np.stack(window), features) - mean) / std
            with torch.no_grad():
                prob = torch.softmax(model(torch.from_numpy(x[None]).float()), 1)[0].numpy()
            act_top = [(actions[i], float(prob[i])) for i in np.argsort(prob)[::-1][:3]]
            if still and STATIC:
                # holding still: a movement class (tennis serve, bowling, waving ...) is impossible
                cand = max(STATIC, key=lambda a: prob[actions.index(a)])
                i = actions.index(cand)
                recent.append(i)
            elif prob.max() >= args.threshold:
                recent.append(int(prob.argmax()))
            if recent:
                idx, votes = Counter(recent).most_common(1)[0]
                # change the label only when the new one wins at least half of the recent votes
                if actions[idx] == activity or votes >= len(recent) // 2 + 1:
                    activity, conf = actions[idx], float(prob[idx])

        now = time.time()
        if now - last_log >= 1.0:
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), activity, f"{conf:.2f}", reps, int(fall), posture])
            log_f.flush()
            last_log = now

        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        if posture:     # a held posture is the main label; the activity model is set aside while still
            cv2.putText(frame, f"Pose: {posture} ({pose_conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 180, 60), 2)
        else:
            cv2.putText(frame, f"{activity} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Squat reps: {reps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if lm is not None and not legs_visible:
            msg = "Step back: show knees and feet"
            cv2.rectangle(frame, (0, 72), (w, 108), (0, 140, 255), -1)
            cv2.putText(frame, msg, (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if args.debug:
            lines = [f"movement {motion:.3f} ({'STILL' if still else 'moving'}, still below {args.still})  "
                     f"legs {'visible' if legs_visible else 'NOT visible'}",
                     "activity: " + ", ".join(f"{n} {v:.2f}" for n, v in act_top),
                     "posture:  " + ", ".join(f"{n} {v:.2f}" for n, v in pose_top)]
            cv2.rectangle(frame, (0, h - 80), (w, h), (30, 30, 30), -1)
            for k, line in enumerate(lines):
                cv2.putText(frame, line, (10, h - 58 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if fall:
            cv2.putText(frame, "FALL DETECTED", (w - 300, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("ActiSense", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
log_f.close()
print(f"Session log saved to {args.log}")
