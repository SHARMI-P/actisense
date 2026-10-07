# ActiSense – Complete Procedure

Commands are shown for Windows (Command Prompt). On Linux/macOS use `python3`, and
`source venv/bin/activate` instead of `venv\Scripts\activate`.

## Phase 0 – Install
Requirements: Python 3.10, 3.11 or 3.12 (NOT 3.13), a webcam, ~6 GB free disk, 8 GB RAM.

```
cd path\to\actisense
python --version
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import mediapipe, torch, cv2; print('OK', mediapipe.__version__)"
```
Expected: `OK 0.10.14`.

How: a virtual environment keeps this project's library versions separate from the rest of your PC.
MediaPipe 0.10.14 is pinned because newer versions removed the `mp.solutions.pose` API the code uses.

Webcam check (should show a skeleton on you; press q):
```
python collect_data.py --action test --sequences 2
rmdir /s /q data\test
```

## Phase 1 – Download datasets
```
python download_datasets.py
```
Creates `videos\<action>\*.avi` (KTH, ~1.1 GB) and `urfall\falls\`, `urfall\adls\` (UR Fall camera-0 RGB frames).
Re-run it if the internet drops; finished files are skipped.
Manual alternative: https://www.csc.kth.se/cvap/actions/ and https://fenix.ur.edu.pl/~mkepski/ds/uf.html
(only the `*-cam0-rgb.zip` files).

## Phase 2 – Build the training data
```
python extract_dataset.py --root videos --map handwaving=waving,handclapping=clapping --skip jogging
```
How: for every frame, MediaPipe finds 33 body keypoints. Each skeleton is normalised (centred on the hips,
scaled by torso length) so position and distance from the camera do not matter. 30 consecutive skeletons
(~1 s) form one training sample, saved to `data\<activity>\`. Small KTH videos are upscaled first.
This is the slow step (rough estimate 1-3 h on a CPU) – run it once, overnight if needed.

Add activities KTH does not have, with your webcam (2-3 m away, whole body visible):
```
python collect_data.py --action sitting   --sequences 60
python collect_data.py --action standing  --sequences 60
python collect_data.py --action squatting --sequences 60
python collect_data.py --action lying     --sequences 60
```
Tip: record 2-3 different people; vary clothes, lighting and angle.

Check counts:
```
python -c "import os; [print(d, len(os.listdir('data/'+d))) for d in sorted(os.listdir('data'))]"
```

## Phase 3 – Train and evaluate the activity model
```
python train.py --test-pattern "person(02|03|05|06|07|08|09|10|22)_"
```
How: a 2-layer LSTM reads each 30-frame sequence and learns which motion pattern belongs to which
activity. KTH people 02,03,05-10,22 (the official KTH test split) are never used for training, so the
reported accuracy is on unseen people. Your webcam clips are split 80/20 automatically.
Outputs: `har_model.pt`, `confusion_matrix.png`, and a precision/recall/F1 table in the terminal
(copy it into your report).

## Phase 4 – Evaluate the fall detector on UR Fall
```
python evaluate_falls.py --root urfall
```
How: the rule flags a fall when the hips drop quickly (within ~0.5 s) AND the torso becomes close to
horizontal. The script runs it on 30 real falls and 40 normal-activity clips for 12 threshold settings
and reports sensitivity (falls caught) and specificity (normal activity not flagged).
Then put the best values into `run_app.py`:
```python
fall_det = FallDetector(drop=0.15, tilt=60)   # use the values the script printed
```
Outputs: `fall_eval.csv`.

## Phase 5 – Live demo and analysis
```
python run_app.py
```
Do several activities, a few squats, and a controlled "fall" onto a mattress. Press q to stop.
How: the last 30 frames go through the trained LSTM every frame; predictions under 60% confidence are
ignored and a vote over the last 8 removes flicker. The squat counter tracks the knee angle
(below 100° = down, above 160° = up). One log row per second goes to `logs\session.csv`.

```
streamlit run dashboard.py
```
Opens a browser page: totals, time per activity, timeline, transitions, confidence. Take screenshots.

## Phase 6 – Optional sensor baseline
Download "UCI HAR Dataset" from the UCI Machine Learning Repository, unzip next to the code:
```
python uci_har_baseline.py "UCI HAR Dataset"
```

## Phase 7 – Novelty and report
Add the chosen novelty (recommended: open-set + few-shot recognition), compare against this baseline.
Report: introduction, related work, datasets (cite KTH: Schuldt et al., ICPR 2004; UR Fall: Kwolek &
Kepski, 2014), method, results (confusion matrix, fall table, dashboard), limitations, future work.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No matching distribution for mediapipe==0.10.14` | You are on Python 3.13. Install 3.11 and recreate the venv |
| `module 'mediapipe' has no attribute 'solutions'` | Wrong MediaPipe version: `pip install mediapipe==0.10.14` |
| `Cannot open source: 0` | Close other camera apps; try `--source 1` |
| Skeleton flickers / not found | More light, step back so feet are visible |
| Low accuracy for one class | Record more varied clips of it, retrain |
| Live label always "unknown" | Model unsure: more data, or `python run_app.py --threshold 0.4` |
| `Need recorded data for at least 2 actions` | Run Phase 2 first; check the data folder |
