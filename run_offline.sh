#!/bin/bash
python main.py \
	--root_path ~/ \
	--video_path /data2/EgoGesture/images \
	--annotation_path ~/Real-time-GesRec/annotation_EgoGesture/egogestureall_but_None.json\
	--result_path ~/Real-time-GesRec/results \
	--dataset egogesture \
	--sample_duration 32 \
    	--learning_rate 0.01 \
    	--model c3d \
	--model_depth 10 \
	--resnet_shortcut B \
	--batch_size 8 \
	--n_classes 83 \
	--n_finetune_classes 83 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality RGB-D \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
     	--n_epochs 100 \

