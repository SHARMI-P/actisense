"""Shared constants, pose-feature extraction and the LSTM model for ActiSense."""
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

SEQ_LEN = 30            # frames per window (about 1 s at 30 fps)
N_FEATURES = 33 * 4     # 33 landmarks x (x, y, z, visibility)
ACTIONS = ["standing", "sitting", "walking", "squatting", "waving", "lying"]

# MediaPipe Pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def landmarks_array(results):
    """Return a (33, 4) array of x, y, z, visibility, or None if no person is detected."""
    if results.pose_landmarks is None:
        return None
    return np.array([[p.x, p.y, p.z, p.visibility]
                     for p in results.pose_landmarks.landmark], dtype=np.float32)


def normalize(lm):
    """Make features independent of where the person stands and how far they are from the camera.
    Coordinates are centred on the mid-hip and scaled by torso length."""
    hip = (lm[L_HIP, :3] + lm[R_HIP, :3]) / 2
    shoulder = (lm[L_SHOULDER, :3] + lm[R_SHOULDER, :3]) / 2
    scale = np.linalg.norm(shoulder[:2] - hip[:2]) + 1e-6
    xyz = (lm[:, :3] - hip) / scale
    return np.concatenate([xyz.flatten(), lm[:, 3]]).astype(np.float32)   # 132 values


def select_features(arr, mode="all"):
    """Pick the per-frame features the model uses.
    'all' = x, y, z of 33 landmarks + 33 visibility scores (132 values)
    'xyz' = coordinates only (99 values). Visibility scores are much lower in small, blurry dataset
    videos than on a sharp webcam, so they can tell the model *which camera* a clip came from;
    dropping them removes that shortcut."""
    return arr[..., :99] if mode == "xyz" else arr


VIDEO_EXT =(".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".webm")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def find_sequences(folder):
    """Every video file, plus every folder of image frames, below `folder` (one item = one clip)."""
    import os
    seqs = []
    for dirpath, _, files in os.walk(folder):
        files = sorted(files)
        seqs += [os.path.join(dirpath, f) for f in files if f.lower().endswith(VIDEO_EXT)]
        if any(f.lower().endswith(IMAGE_EXT) for f in files):
            seqs.append(dirpath)
    return sorted(seqs)


def iter_frames(path, min_side=0):
    """Yield BGR frames from a video file or a folder of image frames (sorted by name)."""
    import os
    import cv2

    def fit(frame):
        h, w = frame.shape[:2]
        if min_side and min(h, w) < min_side:
            s = min_side / min(h, w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        return frame

    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(IMAGE_EXT):
                frame = cv2.imread(os.path.join(path, f))
                if frame is not None:
                    yield fit(frame)
    else:
        cap = cv2.VideoCapture(path)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield fit(frame)
        cap.release()


def joint_angle(a, b, c):
    """Angle ABC in degrees (b is the joint)."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class FallDetector:
    """Rule-based fall detector shared by run_app.py and evaluate_falls.py.

    A fall is flagged when, within `window` frames (~0.5 s), the mid-hip drops by more than
    `drop` (fraction of image height) AND the torso is tilted more than `tilt` degrees from vertical.
    """

    def __init__(self, drop=0.2, tilt=60.0, window=15):
        from collections import deque
        self.drop, self.tilt = drop, tilt
        self.hip_hist = deque(maxlen=window)

    def update(self, lm, w, h):
        """lm: (33, 4) landmarks in normalised image coords. Returns True if a fall is detected now."""
        px = lm[:, :2] * np.array([w, h], dtype=np.float32)
        hip_y = float((lm[L_HIP, 1] + lm[R_HIP, 1]) / 2)
        sh = (px[L_SHOULDER] + px[R_SHOULDER]) / 2
        hp = (px[L_HIP] + px[R_HIP]) / 2
        tilt = np.degrees(np.arctan2(abs(sh[0] - hp[0]), abs(sh[1] - hp[1]) + 1e-6))
        self.hip_hist.append(hip_y)
        full = len(self.hip_hist) == self.hip_hist.maxlen
        return bool(full and hip_y - min(self.hip_hist) > self.drop and tilt > self.tilt)


class HARModel(nn.Module):
    """2-layer LSTM over a window of pose frames, followed by a small classifier head."""

    def __init__(self, n_features, n_classes, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])