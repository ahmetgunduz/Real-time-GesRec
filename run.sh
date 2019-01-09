#!/bin/bash
python main.py \
	--root_path ~/ \
	--video_path Desktop/EgoGesture/images \
	--annotation_path Desktop/Real-time-GesRec/annotation_EgoGesture/egogestureall_but_None.json\
	--result_path Desktop/Real-time-GesRec/results \
	--dataset egogesture \
	--sample_duration 32 \
    --learning_rate 0.01 \
    --model resnetl \
	--model_depth 10 \
	--resnet_shortcut A \
	--batch_size 96 \
	--n_classes 83 \
	--n_finetune_classes 83 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality Depth \
	--train_crop random \
	--n_val_samples 1 \
	--test_subset test \
    --n_epochs 100 \
    --no_cuda \
    
