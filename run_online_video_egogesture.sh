#!/bin/bash

# "$1" test video path

python online_test_video.py \
	--root_path ~/Real-time-GesRec \
	--resume_path_det results/egogesture_resnetl_10_RGB_8.pth \
	--resume_path_clf results/egogesture_resnext_101_RGB_32.pth  \
    --video $1 \
	--sample_duration 8 \
	--sample_duration_det 8 \
	--sample_duration_clf 32 \
	--model_det resnetl \
	--model_clf resnext \
	--model_depth_det 10 \
	--width_mult_det 0.5 \
	--width_mult_clf 1 \
	--model_depth_clf 101 \
	--resnet_shortcut_det A \
    --resnet_shortcut_clf B \
	--batch_size 1 \
	--n_classes_det 2 \
	--n_finetune_classes_det 2 \
	--n_classes_clf 83 \
	--n_finetune_classes_clf 83 \
	--n_threads 16 \
	--modality_det RGB \
	--modality_clf RGB \
	--n_val_samples 1 \
	--det_strategy median \
	--det_queue_size 4 \
	--det_counter 2 \
	--clf_strategy median \
	--clf_queue_size 32 \
	--clf_threshold_pre 1.0 \
	--clf_threshold_final 0.15 \
	--stride_len 1
