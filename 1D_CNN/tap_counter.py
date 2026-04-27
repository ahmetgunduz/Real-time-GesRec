"""
tap_counter.py
──────────────
Online and offline tap counting using a trained Tap1DCNN checkpoint.

Two input modes
───────────────
  Live / video   Opens a webcam or video file, runs MediaPipe hand tracking
                 per frame, extracts features, and feeds a rolling window into
                 the model in real time.

  CSV replay     Reads a previously recorded CSV (from collect_tap_data.py)
                 and replays the feature vectors through the model without
                 any camera or MediaPipe dependency.

Tap detection logic
───────────────────
  A tap event is fired when:
    1. The sigmoid output of the model crosses --threshold (default 0.5), and
    2. At least --debounce_frames have elapsed since the last detected tap
       (default 15 frames ≈ 0.5 s at 30 fps → max 2 taps/sec).

Usage
─────
  # live webcam
  python tap_counter.py --model best_1dcnn_tap.pt

  # pre-recorded video file
  python tap_counter.py --model best_1dcnn_tap.pt --source path/to/clip.mp4

  # CSV replay (no camera needed)
  python tap_counter.py --model best_1dcnn_tap.pt --csv tap_data/session01.csv

  # tune sensitivity
  python tap_counter.py --model best_1dcnn_tap.pt --threshold 0.4 --debounce_frames 12

  # save per-frame inference results
  python tap_counter.py --model best_1dcnn_tap.pt --out_csv results.csv
"""

import argparse
import csv
import math
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import mediapipe as mp


# ── Shared constants (must match collect_tap_data.py / 1dcnn.py) ─────────────
FEATURE_COLUMNS = ["dist_raw", "dist_norm", "velocity", "accel", "hand_conf"]

THUMB_TIP = 4
INDEX_TIP = 8
WRIST     = 0
MID_MCP   = 9

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Path resolution
_SCRIPT_DIR = Path(__file__).parent.resolve()
_WORKSPACE_ROOT = _SCRIPT_DIR.parent  # parent of 1D_CNN/
_DEFAULT_TASK_MODEL_PATH = _WORKSPACE_ROOT / 'models' / 'hand_landmarker.task'

DEFAULT_TASK_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
    'hand_landmarker/float16/1/hand_landmarker.task'
)
DEFAULT_TASK_MODEL_PATH = str(_DEFAULT_TASK_MODEL_PATH)


# ── Model definition (must match 1dcnn.py) ───────────────────────────────────
class Tap1DCNN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_features, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.features(x).pipe(self.classifier).squeeze(1) \
            if False else self.classifier(self.features(x)).squeeze(1)


# ── Checkpoint loading ────────────────────────────────────────────────────────
class InferenceState:
    """Holds the loaded model, scaler parameters, and window config."""

    def __init__(self, model, scaler_mean, scaler_scale, window_size, feature_columns):
        self.model         = model
        self.scaler_mean   = scaler_mean   # np.ndarray [n_features]
        self.scaler_scale  = scaler_scale  # np.ndarray [n_features]
        self.window_size   = window_size
        self.feature_columns = feature_columns

    def scale(self, window: np.ndarray) -> np.ndarray:
        """Standardise a [window_size, n_features] array using saved scaler."""
        return ((window - self.scaler_mean) / self.scaler_scale).astype(np.float32)

    @staticmethod
    def load(checkpoint_path: str) -> "InferenceState":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        feature_columns = ckpt.get("feature_columns", FEATURE_COLUMNS)
        num_features    = len(feature_columns)
        window_size     = ckpt.get("window_size", 30)

        model = Tap1DCNN(num_features)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        return InferenceState(
            model          = model,
            scaler_mean    = np.array(ckpt["scaler_mean"],  dtype=np.float32),
            scaler_scale   = np.array(ckpt["scaler_scale"], dtype=np.float32),
            window_size    = window_size,
            feature_columns= feature_columns,
        )


# ── Inference engine ──────────────────────────────────────────────────────────
class TapInferenceEngine:
    """
    Accepts one feature vector at a time, maintains a rolling window buffer,
    runs the model, and applies debounce logic to fire discrete tap events.
    """

    def __init__(self, state: InferenceState, threshold: float, debounce_frames: int):
        self.state           = state
        self.threshold       = threshold
        self.debounce_frames = debounce_frames

        self._buffer             = deque(maxlen=state.window_size)
        self._frames_since_tap   = debounce_frames  # start ready to detect
        self._tap_count          = 0
        self._last_prob          = 0.0
        self._last_pred          = 0
        self._prev_pred          = 0  # prediction from the previous frame (for edge detection)

    @property
    def tap_count(self):
        return self._tap_count

    @property
    def last_prob(self):
        return self._last_prob

    @property
    def last_pred(self):
        return self._last_pred

    def push(self, feature_vector: np.ndarray) -> bool:
        """
        Push one feature vector (shape [n_features]).
        Returns True if a new tap was just detected this frame.
        """
        self._buffer.append(feature_vector)
        self._frames_since_tap += 1

        if len(self._buffer) < self.state.window_size:
            return False

        window = np.stack(self._buffer)              # [window_size, n_features]
        window = self.state.scale(window)            # standardise
        tensor = torch.tensor(window).unsqueeze(0)  # [1, window_size, n_features]
        tensor = tensor.permute(0, 2, 1)            # [1, n_features, window_size]

        with torch.no_grad():
            logit = self.state.model(tensor)
            prob  = torch.sigmoid(logit).item()

        self._last_prob = prob
        self._last_pred = int(prob >= self.threshold)

        # Rising-edge trigger: only fire on 0→1 transition, not while staying high.
        # This prevents the model's 30-frame positive window from causing double-counts
        # when debounce_frames < window_size.
        tap_fired = False
        rising_edge = (self._last_pred == 1 and self._prev_pred == 0)
        if rising_edge and self._frames_since_tap >= self.debounce_frames:
            self._tap_count      += 1
            self._frames_since_tap = 0
            tap_fired             = True

        self._prev_pred = self._last_pred
        return tap_fired


# ── Feature extraction helpers ────────────────────────────────────────────────

    def reset_on_hand_loss(self):
        """Call when the hand leaves frame.

        Clears the rolling buffer and resets the edge-detector state so that
        a reappearing hand doesn't trigger a spurious rising-edge tap.
        """
        self._buffer.clear()
        self._prev_pred = 0
        self._last_pred = 0
        self._last_prob = 0.0
        # Do NOT reset _tap_count or _frames_since_tap.


# ── Feature extraction helpers ────────────────────────────────────────────────
def _dist(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def ensure_task_model(model_path: str, no_download: bool) -> str:
    if os.path.exists(model_path):
        return model_path
    if no_download:
        raise FileNotFoundError(
            f'Hand model not found: {model_path}. '
            f'Download from: {DEFAULT_TASK_MODEL_URL}'
        )
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    print(f'Downloading hand model to: {model_path}')
    urllib.request.urlretrieve(DEFAULT_TASK_MODEL_URL, model_path)
    return model_path


def create_hand_landmarker(args):
    model_path   = ensure_task_model(args.hand_model, args.no_model_download)
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options      = mp.tasks.vision.HandLandmarkerOptions(
        base_options              = base_options,
        running_mode              = mp.tasks.vision.RunningMode.IMAGE,
        num_hands                 = 1,
        min_hand_detection_confidence = args.min_det_conf,
        min_hand_presence_confidence  = args.min_track_conf,
        min_tracking_confidence       = args.min_track_conf,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def select_hand(results, preference: str):
    if not results.hand_landmarks:
        return None, None
    for i, lm in enumerate(results.hand_landmarks):
        label = 'unknown'
        conf  = 0.0
        if i < len(results.handedness) and results.handedness[i]:
            handed = results.handedness[i][0]
            label  = handed.category_name.lower()
            conf   = handed.score
        if preference == 'any' or label == preference:
            return lm, conf
    return None, None


def extract_features(lm, w: int, h: int, dist_norm_hist: deque) -> np.ndarray:
    """Compute one feature vector from a MediaPipe landmark list."""
    thumb     = lm[THUMB_TIP]
    index_tip = lm[INDEX_TIP]
    wrist_lm  = lm[WRIST]
    mid_mcp   = lm[MID_MCP]

    dist_raw  = _dist(thumb, index_tip, w, h)
    palm_size = _dist(wrist_lm, mid_mcp, w, h)
    dist_norm = dist_raw / palm_size if palm_size > 1e-6 else 0.0

    dist_norm_hist.append(dist_norm)
    vel   = dist_norm_hist[-1] - dist_norm_hist[-2] if len(dist_norm_hist) >= 2 else 0.0
    accel = ((dist_norm_hist[-1] - dist_norm_hist[-2]) -
             (dist_norm_hist[-2] - dist_norm_hist[-3])) if len(dist_norm_hist) >= 3 else 0.0

    return np.array([dist_raw, dist_norm, vel, accel, 0.0], dtype=np.float32)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_skeleton(frame, lm, w, h):
    for i, j in HAND_CONNECTIONS:
        xi, yi = int(lm[i].x * w), int(lm[i].y * h)
        xj, yj = int(lm[j].x * w), int(lm[j].y * h)
        cv2.line(frame, (xi, yi), (xj, yj), (0, 220, 220), 1)
    for p in lm:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 3, (255, 255, 255), -1)


def draw_overlay(frame, engine: TapInferenceEngine, fps_str: str, hand_visible: bool):
    h, w = frame.shape[:2]

    tap_colour  = (0, 255, 80) if engine.last_pred else (200, 200, 200)
    count_str   = f'Taps: {engine.tap_count}'
    prob_str    = f'Prob: {engine.last_prob:.2f}'
    hand_str    = 'Hand: visible' if hand_visible else 'Hand: not detected'

    cv2.putText(frame, fps_str,   (8, 22),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, hand_str,  (8, 44),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, prob_str,  (8, 66),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, count_str, (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,  tap_colour,      2, cv2.LINE_AA)

    # Confidence bar (full width at bottom)
    bar_w = int(engine.last_prob * w)
    bar_colour = (0, 200, 80) if engine.last_pred else (60, 60, 200)
    cv2.rectangle(frame, (0, h - 12), (bar_w, h), bar_colour, -1)
    cv2.putText(frame, 'Q=quit', (w - 72, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


def open_capture(source: str):
    if source.isdigit():
        idx = int(source)
        if sys.platform == 'darwin' and hasattr(cv2, 'CAP_AVFOUNDATION'):
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
        if hasattr(cv2, 'CAP_V4L2'):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(idx)
    return cv2.VideoCapture(source)


# ── CSV replay mode ───────────────────────────────────────────────────────────
def run_csv_mode(args, engine: TapInferenceEngine):
    """
    Replay a pre-recorded CSV through the model.
    Compares predicted taps to ground-truth labels if present.
    """
    df = pd.read_csv(args.csv)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f'ERROR: CSV is missing columns: {missing}')

    has_gt       = 'label' in df.columns
    out_rows     = []
    tap_frames   = []

    print(f'\nReplaying {len(df)} frames from {args.csv}')
    if has_gt:
        gt_taps = int(df['label'].diff().clip(lower=0).sum())
        print(f'Ground-truth tap transitions in CSV: {gt_taps}')

    for row in df.itertuples(index=False):
        fvec = np.array([getattr(row, c) for c in FEATURE_COLUMNS], dtype=np.float32)
        tap_fired = engine.push(fvec)

        out_row = {
            'frame_idx': getattr(row, 'frame_idx', len(out_rows)),
            'prob':      round(engine.last_prob, 4),
            'pred':      engine.last_pred,
            'tap_count': engine.tap_count,
        }
        if has_gt:
            out_row['gt_label'] = int(getattr(row, 'label', 0))
        if tap_fired:
            tap_frames.append(out_row['frame_idx'])
        out_rows.append(out_row)

    print(f'\nDetected {engine.tap_count} taps at frames: {tap_frames}')

    if args.out_csv:
        _write_csv(args.out_csv, out_rows)

    if has_gt and out_rows:
        gt_labels = np.array([r['gt_label'] for r in out_rows])
        pred_preds = np.array([r['pred']    for r in out_rows])
        _print_accuracy(gt_labels, pred_preds)

    if args.plot_conf and out_rows:
        _plot_confidence(out_rows)


# ── Live / video mode ─────────────────────────────────────────────────────────
def run_video_mode(args, engine: TapInferenceEngine):
    cap = open_capture(args.source)
    if not cap.isOpened():
        sys.exit(f'ERROR: Could not open video source: {args.source}')

    hand_landmarker = create_hand_landmarker(args)
    dist_norm_hist  = deque(maxlen=3)

    fps_counter = 0
    fps_t0      = time.time()
    fps_str     = ''
    out_rows    = []
    frame_idx   = 0
    hand_was_visible = False

    print(f'\nStarting live inference — Q to quit')
    print(f'Threshold={args.threshold}  Debounce={args.debounce_frames} frames\n')

    # main live video loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of stream.')
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        t0 = time.time()

        # MediaPipe inference
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = hand_landmarker.detect(mp_image)

        lm, hand_conf = select_hand(results, args.hand)

        tap_fired = False
        if lm is not None:
            fvec         = extract_features(lm, w, h, dist_norm_hist)
            fvec[4]      = float(hand_conf)   # fill hand_conf slot
            tap_fired    = engine.push(fvec)
            draw_skeleton(frame, lm, w, h)

            # Thumb/index tap line
            t_lm  = lm[THUMB_TIP]
            i_lm  = lm[INDEX_TIP]
            tx, ty   = int(t_lm.x * w), int(t_lm.y * h)
            ix_, iy  = int(i_lm.x * w), int(i_lm.y * h)
            line_col = (0, 0, 220) if engine.last_pred else (0, 200, 0)
            cv2.line(frame, (tx, ty), (ix_, iy), line_col, 2)
        else:
            if hand_was_visible:
                # Hand just left frame — reset buffer and edge state so that
                # the reappearing hand doesn't fire a spurious rising-edge tap.
                engine.reset_on_hand_loss()
                dist_norm_hist.clear()

        hand_was_visible = lm is not None

        if tap_fired:
            print(f'  TAP #{engine.tap_count}  frame={frame_idx}  prob={engine.last_prob:.3f}')

        draw_overlay(frame, engine, fps_str, lm is not None)
        cv2.imshow('Tap Counter', frame)

        if args.out_csv:
            out_rows.append({
                'frame_idx': frame_idx,
                'prob':      round(engine.last_prob, 4),
                'pred':      engine.last_pred,
                'tap_count': engine.tap_count,
                'tap_fired': int(tap_fired),
            })

        fps_counter += 1
        frame_idx   += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps_str     = f'{fps_counter / elapsed:.1f} FPS'
            fps_counter = 0
            fps_t0      = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()

    print(f'\nTotal taps detected: {engine.tap_count}')

    if args.out_csv and out_rows:
        _write_csv(args.out_csv, out_rows)

    if args.plot_conf and out_rows:
        _plot_confidence(out_rows)


# ── Utility ───────────────────────────────────────────────────────────────────
def _write_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved inference CSV → {path}')


def _print_accuracy(gt: np.ndarray, pred: np.ndarray):
    tp = int(np.sum((gt == 1) & (pred == 1)))
    fp = int(np.sum((gt == 0) & (pred == 1)))
    fn = int(np.sum((gt == 1) & (pred == 0)))
    tn = int(np.sum((gt == 0) & (pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    acc       = (tp + tn) / len(gt) if len(gt) > 0 else 0.0
    print(f'\nFrame-level accuracy vs ground-truth labels:')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall:    {recall:.4f}')
    print(f'  F1:        {f1:.4f}')
    print(f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}')


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Tap counter — online and CSV replay modes.')

    # Model
    p.add_argument('--model',   required=True,
                   help='Path to trained .pt checkpoint (from 1dcnn.py)')

    # Input — mutually exclusive: live/video OR csv replay
    src = p.add_mutually_exclusive_group()
    src.add_argument('--source', default='0',
                     help='Camera index or video file path (default: 0)')
    src.add_argument('--csv',   default=None,
                     help='Path to a CSV file for offline replay')

    # Inference tuning
    p.add_argument('--threshold',       type=float, default=0.7,
                   help='Sigmoid probability threshold for a positive tap prediction (default: 0.5)')
    p.add_argument('--debounce_frames', type=int,   default=15,
                   help='Minimum frames between tap events (default: 15 ≈ 0.5 s at 30 fps → max 2 taps/sec)')

    # MediaPipe (live/video mode only)
    p.add_argument('--hand',           choices=['left', 'right', 'any'], default='any')
    p.add_argument('--mirror',         action='store_true',
                   help='Horizontally flip the webcam feed')
    p.add_argument('--hand_model',     default=DEFAULT_TASK_MODEL_PATH,
                   help='Path to hand_landmarker.task')
    p.add_argument('--no_model_download', action='store_true')
    p.add_argument('--min_det_conf',   type=float, default=0.6)
    p.add_argument('--min_track_conf', type=float, default=0.5)

    # Output
    p.add_argument('--out_csv', default=None,
                   help='Optional path to save per-frame inference results')

    p.add_argument('--plot_conf', action='store_true',
               help='Plot confidence over frames after run')

    return p.parse_args()



def _plot_confidence(rows, out_path='airtap/outputs/confidence_plot.png'):
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    frame_idxs = [r['frame_idx'] for r in rows]
    probs      = [r['prob'] for r in rows]

    plt.figure()
    plt.plot(frame_idxs, probs)
    plt.xlabel('Frame Index')
    plt.ylabel('Confidence (prob)')
    plt.title('Confidence Over Frames')

    plt.savefig(out_path)
    plt.close()

    print(f'Saved confidence plot → {out_path}')


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    state  = InferenceState.load(args.model)
    engine = TapInferenceEngine(state, args.threshold, args.debounce_frames)

    print(f'Loaded checkpoint: {args.model}')
    print(f'Window size: {state.window_size}  Features: {state.feature_columns}')
    print(f'Threshold: {args.threshold}  Debounce: {args.debounce_frames} frames')

    if args.csv:
        run_csv_mode(args, engine)
    else:
        run_video_mode(args, engine)


if __name__ == '__main__':
    main()
