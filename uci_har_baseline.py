"""Sensor-based baseline on the UCI HAR smartphone dataset (for the comparison chapter of the report).

python uci_har_baseline.py "UCI HAR Dataset"
"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

root = sys.argv[1] if len(sys.argv) > 1 else "UCI HAR Dataset"


def load(p):
    return np.loadtxt(os.path.join(root, p))


Xtr, ytr = load("train/X_train.txt"), load("train/y_train.txt").astype(int)
Xte, yte = load("test/X_test.txt"), load("test/y_test.txt").astype(int)
with open(os.path.join(root, "activity_labels.txt")) as f:
    labels = [line.split()[1] for line in f.read().strip().splitlines()]

print(f"Train {Xtr.shape}, test {Xte.shape} (official split: different subjects in train and test)")
models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000)),
}
for name, clf in models.items():
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"\n=== {name}: accuracy {accuracy_score(yte, pred):.4f} ===")
    print(classification_report(yte, pred, target_names=labels))
