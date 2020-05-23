#!/bin/bash

# "$1" classıfıer resume path
# "$2" model_clf
# "$3" width_mult
# "$4" classıfıer modalıty
python online_test.py \
	--root_path ~/\
	--video_path datasets/nvGesture \
	--annotation_path Efficient-3DCNNs-Ahmet/annotation_nvGesture/nvall.json \
	--resume_path_det Efficient-3DCNNs-Ahmet/results/shared/nv_resnetl_10_Depth_8.pth \
	--resume_path_clf "$1"  \
	--result_path Efficient-3DCNNs-Ahmet/results \
	--dataset nvgesture    \
	--sample_duration_det 8 \
	--sample_duration_clf 32 \
	--model_det resnetl \
	--model_clf "$2" \
	--model_depth_det 10 \
	--width_mult_det 0.5 \
	--model_depth_clf 101 \
	--width_mult_clf "$3" \
	--resnet_shortcut_det A \
	--resnet_shortcut_clf B \
	--batch_size 1 \
	--n_classes_det 2 \
	--n_finetune_classes_det 2 \
	--n_classes_clf 25 \
	--n_finetune_classes_clf 25 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality_det Depth \
	--modality_clf "$4" \
	--n_val_samples 1 \
	--train_crop random \
	--test_subset test  \
	--det_strategy median \
	--det_queue_size 4 \
	--det_counter 2 \
	--clf_strategy median \
	--clf_queue_size 16 \
	--clf_threshold_pre 1.0 \
	--clf_threshold_final 0.15 \
	--stride_len 1 \