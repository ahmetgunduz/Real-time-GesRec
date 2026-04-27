"""
collect_tap_data.py
───────────────────
Real-time pointer-thumb tap data collection using MediaPipe Hands.

Controls
────────
  SPACE  hold to mark frames as tap active (label = 1)
  S start / resume recording (default: starts immediately)
  P pause recording without quitting
  Q quit and save CSV

Output CSV columns
──────────────────
    session_id   recording session identifier (same for all rows in one run)
  frame_idx    monotone frame counter
  timestamp    seconds since start
  dist_raw     Euclidean distance (in image pixels) between thumb tip (4)
                and index finger tip (8)
  dist_norm    dist_raw normalised by palm size
                (distance between wrist (0) and middle-finger MCP (9))
  velocity     finite-difference of dist_norm over 1 frame
  accel        finite-difference of velocity over 1 frame
  hand_conf    MediaPipe detection confidence for the tracked hand
  label        1 while spacebar is held, 0 otherwise

Usage
─────
  python collect_tap_data.py                         # webcam, saves tap_data.csv
  python collect_tap_data.py --out my_session.csv
  python collect_tap_data.py --source /path/to.mp4  # pre-recorded video
  python collect_tap_data.py --source 1              # camera index 1
  python collect_tap_data.py --hand left             # track left hand
  python collect_tap_data.py --mirror                # flip webcam image
    python collect_tap_data.py --session_id s01        # explicit session id
"""

import argparse
import csv
import math
import os
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp


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


# ── Landmark indices of interest ─────────────────────────────────────────────
THUMB_TIP   = 4
INDEX_TIP   = 8
WRIST       = 0
MID_MCP     = 9   # middle finger metacarpophalangeal joint ≈ palm centre


def _dist(a, b, w, h):
    """Euclidean distance between two normalised landmarks in pixel space."""
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def parse_args():
    p = argparse.ArgumentParser(description='Collect pointer-thumb tap data.')
    p.add_argument('--source', default='0',
                   help='Camera index (int) or video file path (default: 0)')
    p.add_argument('--out', default='tap_data.csv',
                   help='Output CSV path (default: tap_data.csv)')
    p.add_argument('--hand', choices=['left', 'right', 'any'], default='any',
                   help='Which hand to track (default: any = first detected)')
    p.add_argument('--mirror', action='store_true',
                   help='Horizontally flip the webcam feed (selfie view)')
    p.add_argument('--max_hands', type=int, default=1,
                   help='Max hands for MediaPipe to detect (default: 1)')
    p.add_argument('--min_det_conf', type=float, default=0.6,
                   help='MediaPipe min detection confidence (default: 0.6)')
    p.add_argument('--min_track_conf', type=float, default=0.5,
                   help='MediaPipe min tracking confidence (default: 0.5)')
    p.add_argument('--paused', action='store_true',
                   help='Start in paused state (press S to begin recording)')
    p.add_argument('--hand_model', default=DEFAULT_TASK_MODEL_PATH,
                   help='Path to hand_landmarker.task model file')
    p.add_argument('--no_model_download', action='store_true',
                   help='Do not auto-download hand_landmarker.task if missing')
    p.add_argument('--session_id', default=None,
                   help='Optional session id to store in CSV (default: inferred from --out)')
    return p.parse_args()


def open_capture(source: str):
    """Open a cv2.VideoCapture from a string that is either a digit or a path."""
    if source.isdigit():
        idx = int(source)
        if sys.platform == 'darwin' and hasattr(cv2, 'CAP_AVFOUNDATION'):
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap

        # On Linux, try V4L2 first to avoid FFMPEG camera enumeration issues
        if hasattr(cv2, 'CAP_V4L2'):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def ensure_task_model(model_path: str, no_download: bool):
    """Ensure the MediaPipe Tasks hand model exists, optionally downloading it."""
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
    model_path = ensure_task_model(args.hand_model, args.no_model_download)

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_det_conf,
        min_hand_presence_confidence=args.min_track_conf,
        min_tracking_confidence=args.min_track_conf,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def select_hand(results, preference: str):
    """Return the landmark list of the preferred hand, or None."""
    if not results.hand_landmarks:
        return None, None

    for i, lm in enumerate(results.hand_landmarks):
        label = 'unknown'
        conf = 0.0
        if i < len(results.handedness) and results.handedness[i]:
            handed = results.handedness[i][0]
            label = handed.category_name.lower()
            conf = handed.score
        if preference == 'any' or label == preference:
            return lm, conf

    return None, None


def draw_hand_skeleton(frame, lm, w, h):
    """Draw hand landmarks and skeletal connections for MediaPipe Tasks output."""
    for i, j in HAND_CONNECTIONS:
        xi, yi = int(lm[i].x * w), int(lm[i].y * h)
        xj, yj = int(lm[j].x * w), int(lm[j].y * h)
        cv2.line(frame, (xi, yi), (xj, yj), (0, 220, 220), 1)

    for p in lm:
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)


def draw_overlay(frame, lm, h, w, dist_raw, dist_norm, vel, accel,
                 label, recording, paused, frame_idx, fps_str):
    """Draw landmarks, distance line, and HUD onto the frame (in-place)."""
    draw_hand_skeleton(frame, lm, w, h)

    # Thumb tip and index tip pixel coords
    t  = lm[THUMB_TIP]
    ix = lm[INDEX_TIP]
    tx, ty = int(t.x * w),  int(t.y * h)
    ix_, iy = int(ix.x * w), int(ix.y * h)

    # Line colour: red when label=1 (tap active), green otherwise
    line_colour = (0, 0, 220) if label else (0, 200, 0)
    cv2.line(frame, (tx, ty), (ix_, iy), line_colour, 2)
    cv2.circle(frame, (tx, ty),  6, (255, 100,   0), -1)
    cv2.circle(frame, (ix_, iy), 6, (  0, 100, 255), -1)

    # HUD
    state_str = 'TAP ACTIVE' if label else 'no tap'
    rec_str   = 'PAUSED' if paused else ('REC' if recording else 'STOPPED')
    hud_lines = [
        fps_str,
        f'Frame: {frame_idx}',
        f'dist_raw:  {dist_raw:.1f} px',
        f'dist_norm: {dist_norm:.4f}',
        f'velocity:  {vel:.4f}',
        f'accel:     {accel:.4f}',
        f'[{rec_str}]  {state_str}',
    ]
    for i, line in enumerate(hud_lines):
        colour = (0, 60, 255) if 'TAP' in line else (200, 200, 200)
        if 'REC' in line:
            colour = (0, 80, 255)
        if 'PAUSED' in line:
            colour = (0, 200, 255)
        cv2.putText(frame, line, (8, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

    # Controls reminder at bottom
    cv2.putText(frame,
                'SPACE=tap  S=rec  P=pause  Q=quit',
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (160, 160, 160), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    # Add timestamp to default output to prevent file overwrites
    if args.out == 'tap_data.csv':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out = f'tap_data/tap_data_{timestamp}.csv'

    session_id = args.session_id
    if session_id is None or str(session_id).strip() == '':
        session_id = os.path.splitext(os.path.basename(args.out))[0]

    cap = open_capture(args.source)
    if not cap.isOpened():
        sys.exit(f'ERROR: Could not open video source "{args.source}"')

    hand_landmarker = create_hand_landmarker(args)

    # ── State ─────────────────────────────────────────────────────────────────
    recording  = not args.paused
    paused     = args.paused
    frame_idx  = 0
    rows       = []
    start_time = time.time()

    # Sliding history for velocity / acceleration (normalised distance)
    hist = deque(maxlen=3)   # stores last 3 dist_norm values
    vel  = 0.0
    accel = 0.0

    fps_counter = 0
    fps_t0      = time.time()
    fps_str     = ''

    print('─' * 50)
    print(f'Collecting to: {args.out}')
    print(f'Session ID: {session_id}')
    print('Controls: SPACE=tap label  S=start/stop rec  P=pause  Q=quit')
    print('─' * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of stream or camera disconnected.')
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # ── MediaPipe inference ───────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = hand_landmarker.detect(mp_image)

        # ── Key handling (non-blocking) ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        label = 0

        if key == ord('q'):
            break
        elif key == ord('s'):
            if paused:
                paused    = False
                recording = True
            else:
                recording = not recording
        elif key == ord('p'):
            paused = not paused
            if paused:
                recording = False

        # Space held = tap active
        # cv2.waitKey captures the key on press; for "hold" detection we check
        # if the key reported is space. To detect holding, poll every frame.
        if key == ord(' '):
            label = 1

        # cv2.waitKey(1) only gives us the most recent press → to simulate
        # "use space to toggle", we query the keyboard state via a small trick:
        # re-poll with waitKey(0) for 1 ms — but that blocks. Instead we rely
        # on the user pressing space EACH frame (common at 15-30 fps it works),
        # OR they can use an alternate approach: toggle mode.
        # We implement TOGGLE mode: first space press turns tap ON, second OFF.
        # This is much more ergonomic for collection.
        # ── toggled state is tracked here ──
        if not hasattr(main, '_tap_on'):
            main._tap_on = False
        if key == ord(' '):
            main._tap_on = not main._tap_on
        label = 1 if main._tap_on else 0

        # ── Feature computation ───────────────────────────────────────────────
        lm, hand_conf = select_hand(results, args.hand)

        dist_raw  = 0.0
        dist_norm = 0.0

        if lm is not None:
            thumb     = lm[THUMB_TIP]
            index_tip = lm[INDEX_TIP]
            wrist     = lm[WRIST]
            mid_mcp   = lm[MID_MCP]

            dist_raw  = _dist(thumb, index_tip, w, h)
            palm_size = _dist(wrist, mid_mcp, w, h)
            dist_norm = dist_raw / palm_size if palm_size > 1e-6 else 0.0

            hist.append(dist_norm)
            if len(hist) >= 2:
                vel = hist[-1] - hist[-2]
            if len(hist) >= 3:
                accel = (hist[-1] - hist[-2]) - (hist[-2] - hist[-3])

            draw_overlay(frame, lm, h, w, dist_raw, dist_norm, vel, accel,
                         label, recording, paused, frame_idx, fps_str)
        else:
            hand_conf = 0.0
            cv2.putText(frame, 'No hand detected', (8, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        # ── Record row ────────────────────────────────────────────────────────
        if recording and not paused and lm is not None:
            rows.append({
                'session_id': session_id,
                'frame_idx':  frame_idx,
                'timestamp':  round(time.time() - start_time, 4),
                'dist_raw':   round(dist_raw,  4),
                'dist_norm':  round(dist_norm, 6),
                'velocity':   round(vel,        6),
                'accel':      round(accel,       6),
                'hand_conf':  round(float(hand_conf), 4),
                'label':      label,
            })

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_counter += 1
        frame_idx   += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            fps_str     = f'{fps_counter / elapsed:.1f} FPS'
            fps_counter = 0
            fps_t0      = time.time()

        cv2.imshow('Tap Data Collection', frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    hand_landmarker.close()
    cv2.destroyAllWindows()

    if not rows:
        print('No data recorded — exiting without writing CSV.')
        return

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fieldnames = ['session_id', 'frame_idx', 'timestamp', 'dist_raw', 'dist_norm',
                  'velocity', 'accel', 'hand_conf', 'label']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total   = len(rows)
    n_tap   = sum(r['label'] for r in rows)
    n_notap = total - n_tap
    print(f'\nSaved {total} frames → {out_path}')
    print(f'  tap (label=1): {n_tap}  ({100*n_tap/total:.1f}%)')
    print(f'  no-tap (label=0): {n_notap}  ({100*n_notap/total:.1f}%)')


if __name__ == '__main__':
    main()
