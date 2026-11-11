"""Shared constants, pose-feature extraction and the LSTM model for ActiSense."""
import numpy as np
import torch
import torch.nn as nn

try:                                   # MediaPipe is only needed for extraction and the live app
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
except (ImportError, AttributeError):  # training on pre-extracted windows works without it
    mp = mp_pose = mp_draw = None

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


# ---------------- static postures (yoga, Bharatanatyam) ----------------
# MediaPipe left/right landmark pairs, used to mirror a pose (a left-leg tree pose = a right-leg one)
LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
            (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
# joint angles that describe a posture: (a, joint, c)
POSE_ANGLES = [(11, 13, 15), (12, 14, 16),     # elbows
               (13, 11, 23), (14, 12, 24),     # shoulders
               (11, 23, 25), (12, 24, 26),     # hips
               (23, 25, 27), (24, 26, 28)]     # knees
N_POSE_FEATURES = 99 + len(POSE_ANGLES)


def mirror_xyz(xyz):
    """Mirror normalised (..., 33, 3) coordinates left <-> right."""
    out = xyz.copy()
    for a, b in LR_PAIRS:
        out[..., a, :], out[..., b, :] = xyz[..., b, :], xyz[..., a, :]
    out[..., 0] = -out[..., 0]
    return out


# extra angles (v5): ankles, and how far torso and thighs lean from vertical - these separate
# standing (all upright), sitting (thighs level) and lying (torso level) directly
EXTRA_ANGLES = [(25, 27, 31), (26, 28, 32)]          # ankles (knee, ankle, foot)
TILT_SEGMENTS = [((23, 24), (11, 12)),                # torso: mid-hip -> mid-shoulder
                 ((23,), (25,)), ((24,), (26,))]      # thighs: hip -> knee
N_POSE_FEATURES_V5 = N_POSE_FEATURES + len(EXTRA_ANGLES) + len(TILT_SEGMENTS)


def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi


def pose_vector(norm, extended=False):
    """One frame -> posture features: 99 normalised coordinates + 8 joint angles (0..1).
    extended=True (v5 models) adds 2 ankle angles + torso and thigh tilt from vertical (112 values).
    `norm` = the 132 values from normalize(), or already (33, 3) coordinates."""
    xyz = np.asarray(norm, dtype=np.float32)
    xyz = xyz.reshape(33, 3) if xyz.shape == (33, 3) else xyz[..., :99].reshape(33, 3)
    ang = [_angle(xyz[a], xyz[b], xyz[c]) for a, b, c in POSE_ANGLES]
    if extended:
        ang += [_angle(xyz[a], xyz[b], xyz[c]) for a, b, c in EXTRA_ANGLES]
        for start, end in TILT_SEGMENTS:
            v = xyz[list(end), :2].mean(0) - xyz[list(start), :2].mean(0)
            ang.append(_angle(np.array([0.0, -1.0]), np.zeros(2), v))   # image y points down: up = (0, -1)
    return np.concatenate([xyz.flatten(), np.array(ang, dtype=np.float32)]).astype(np.float32)


def pose_features_for(n_features):
    """The pose_vector variant a saved posture model was trained with (107 = v4, 112 = v5)."""
    return lambda norm: pose_vector(norm, extended=(n_features == N_POSE_FEATURES_V5))


def motion_energy(frames):
    """Mean frame-to-frame movement of the normalised keypoints (torso lengths per frame).
    Small = the person is holding still (a posture); large = moving (an activity)."""
    f = np.asarray(frames, dtype=np.float32)[..., :99]
    if len(f) < 2:
        return 0.0
    return float(np.abs(np.diff(f, axis=0)).mean())


class PoseMLP(nn.Module):
    """Small classifier for held postures from ONE frame of posture features."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


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

# ---------------- v5: motion features and stronger sequence models ----------------
LR_PERM = list(range(33))
for _a, _b in LR_PAIRS:
    LR_PERM[_a], LR_PERM[_b] = _b, _a


def add_velocity(x):
    """(..., T, F) frames -> (..., T, F + 99): appends each joint's frame-to-frame movement (x, y, z).
    Positions say WHERE the joints are; velocity says HOW they move (direction and speed), which separates
    waving from clapping, running from walking and standing still from everything else.
    Works on numpy arrays and torch tensors; the first frame gets zero velocity."""
    xyz = x[..., :99]
    if isinstance(x, torch.Tensor):
        v = torch.diff(xyz, dim=-2, prepend=xyz[..., :1, :])
        return torch.cat([x, v], -1)
    v = np.diff(xyz, axis=-2, prepend=xyz[..., :1, :])
    return np.concatenate([x, v], -1).astype(np.float32)


def sequence_features(raw, features="all", velocity=False):
    """Raw normalised windows (..., T, 132) -> model input, exactly as in training."""
    x = select_features(raw, features)
    return add_velocity(x) if velocity else x


class BiLSTMAttn(nn.Module):
    """Bidirectional 2-layer LSTM; attention pooling picks the most informative frames of the window."""

    def __init__(self, n_features, n_classes, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.3,
                            bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        out, _ = self.lstm(x)                       # (B, T, 2H)
        w = torch.softmax(self.att(out), dim=1)     # (B, T, 1) weight per frame
        return self.head((w * out).sum(1))


class TCN(nn.Module):
    """Temporal convolution network: 1-D convolutions over time with growing dilation (1, 2, 4),
    residual connections and batch norm; fast on a CPU and strong on skeleton sequences."""

    def __init__(self, n_features, n_classes, hidden=128, kernel=5, dropout=0.3):
        super().__init__()
        self.inp = nn.Conv1d(n_features, hidden, 1)
        self.blocks = nn.ModuleList()
        for d in (1, 2, 4):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=d * (kernel - 1) // 2, dilation=d),
                nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel, padding=d * (kernel - 1) // 2, dilation=d),
                nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)))
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, n_classes))

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))             # (B, H, T)
        for block in self.blocks:
            h = h + block(h)
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1))


MODELS = {"lstm": HARModel, "bilstm": BiLSTMAttn, "tcn": TCN}


def make_model(kind, n_features, n_classes, hidden=128):
    return MODELS[kind](n_features, n_classes, hidden=hidden)


def load_activity_model(path):
    """Load har_model.pt of any version -> (model, info dict with actions, mean, std, features, velocity)."""
    ck = torch.load(path, map_location="cpu")
    info = {"actions": ck["actions"], "mean": np.array(ck["mean"], dtype=np.float32),
            "std": np.array(ck["std"], dtype=np.float32), "features": ck.get("features", "all"),
            "velocity": ck.get("velocity", False), "model": ck.get("model", "lstm")}
    model = make_model(info["model"], len(info["mean"]), len(info["actions"]), ck.get("hidden", 128))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, info
