#!/bin/bash
python main.py \
	--root_path ~/ \
	--video_path /data2/EgoGesture/images \
	--annotation_path ~/Real-time-GesRec/annotation_EgoGesture/egogesturebinary.json\
	--result_path ~/Real-time-GesRec/results \
	--resume_path MyRes3D-Ahmet/report/egogesture_resnetl_10_Depth_8_9939.pth \
	--dataset egogesture \
	--sample_duration 8 \
    --learning_rate 0.01 \
    --model resnetl \
	--model_depth 10 \
	--resnet_shortcut A \
	--batch_size 8 \
	--n_classes 2 \
	--n_finetune_classes 2 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality Depth \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
     --n_epochs 100 \

