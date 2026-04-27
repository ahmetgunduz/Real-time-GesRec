#!/bin/bash
# Run the two-stage (detector + classifier) gesture recognition pipeline
# on a video file or live webcam using Jester RGB weights.
#
# Usage:
#   ./run_online_jester_video.sh                   # live webcam
#   ./run_online_jester_video.sh /path/to/clip.mp4 # pre-recorded video
#
# ── Set these two paths to your downloaded weight files ──────────────────────
DET_WEIGHTS="/path/to/resnet10_jester_or_egogesture.pth"   # TODO: fill in
CLF_WEIGHTS="/path/to/resnext101_jester.pth"               # TODO: fill in
# ─────────────────────────────────────────────────────────────────────────────

# Use first argument as video source; default to webcam (0)
VIDEO="${1:-0}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 online_test_video.py \
    --root_path "" \
    --video         "${VIDEO}" \
    --annotation_path "${REPO_DIR}/annotation_Jester/jester.json" \
    --result_path   "${REPO_DIR}/results" \
    \
    --resume_path_det "${DET_WEIGHTS}" \
    --model_det       resnet \
    --model_depth_det 10 \
    --resnet_shortcut_det A \
    --modality_det    RGB \
    --n_classes_det   2 \
    --n_finetune_classes_det 2 \
    --sample_duration_det    8 \
    \
    --resume_path_clf "${CLF_WEIGHTS}" \
    --model_clf       resnext \
    --model_depth_clf 101 \
    --resnet_shortcut_clf B \
    --modality_clf    RGB \
    --n_classes_clf   27 \
    --n_finetune_classes_clf 27 \
    --sample_duration_clf    16 \
    \
    --batch_size   1 \
    --n_threads    4 \
    --norm_value   1 \
    --mean_dataset kinetics \
    \
    --det_strategy  median \
    --det_queue_size 4 \
    --det_counter    2 \
    --clf_strategy  median \
    --clf_queue_size 8 \
    --clf_threshold_pre   0.6 \
    --clf_threshold_final 0.15 \
    --stride_len    1 \
    --no_cuda

# ── Notes ────────────────────────────────────────────────────────────────────
# n_classes_det=2   → binary detector (gesture / no-gesture).
#   If your ResNet-10 weights were trained on Jester directly (27 classes)
#   rather than as a binary detector, change n_classes_det/n_finetune_classes_det
#   to 27.  The "No_gesture" class (index 2) then acts as the soft negative.
#
# For Apple Silicon: remove --no_cuda to let the script pick MPS automatically.
#
# For EgoGesture RGB det+clf weights (the Google Drive bundle in the repo):
#   --n_classes_det 2  --sample_duration_det 8  --model_det resnetl
#   --n_classes_clf 83 --sample_duration_clf 32 --model_clf resnext
# ─────────────────────────────────────────────────────────────────────────────
