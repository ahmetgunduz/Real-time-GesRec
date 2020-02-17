# "$1" pretrain path
# "$2" model
# "$3" width_mult

python main.py --root_path ~/Desktop/git/ \
	--video_path ~/datasets/EgoGesture \
	--annotation_path Real-time-GesRec/annotation_EgoGesture/egogestureall.json \
	--result_path Real-time-GesRec/results/EgoGesture_RGB_all \
	--pretrain_path "$1" \
	--dataset egogesture \
	--n_classes 27 \
	--n_finetune_classes 84 \
	--model "$2" \
	--groups 3 \
	--version 1.1 \
	--width_mult "$3" \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 16 \
	--modality RGB \
	--downsample 2 \
	--batch_size 80 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
    --n_epochs 20 \
    --train_validate \
    --test_subset test \
    --ft_portion last_layer  \
	# --no_train \
 	# --no_val \
 	# --test
 	# --resume_path Real-time-GesRec/results/egogesture_shufflenet_2.0x_Depth_16_checkpoint.pth \
 		# --pretrain_path Real-time-GesRec/results/jester_mobilenetv2_1.0x_RGB_16_best.pth \