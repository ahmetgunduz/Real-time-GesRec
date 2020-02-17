# "$1" pretrain path
# "$2" model
# "$3" width_mult

python main.py --root_path ~/ \
	--video_path ~/datasets/nvGesture \
	--annotation_path Efficient-3DCNNs/annotation_nvGesture/nvall_but_None.json \
	--result_path Efficient-3DCNNs/results/nvGesture_RGB_all \
	--pretrain_path "$1" \
	--dataset nvgesture \
	--n_classes 27 \
	--n_finetune_classes 26 \
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
    --n_epochs 30 \
    --test_subset test \
    --ft_portion last_layer 
	# --no_train \
 	# --no_val \
 	# --test
 	# --resume_path Efficient-3DCNNs/results/egogesture_shufflenet_2.0x_Depth_16_checkpoint.pth \
 		# --pretrain_path Efficient-3DCNNs/results/jester_mobilenetv2_1.0x_RGB_16_best.pth \




# # "$1" pretrain path
# # "$2" resume path
# # "$3" model
# # "$4" width_mult

# python main.py --root_path ~/ \
# 	--video_path ~/datasets/nvGesture \
# 	--annotation_path Efficient-3DCNNs/annotation_nvGesture/nvall_but_None.json \
# 	--result_path Efficient-3DCNNs/results \
# 	--pretrain_path "$1" \
# 	--resume_path "$2" \
# 	--dataset nvgesture \
# 	--n_classes 27 \
# 	--n_finetune_classes 25 \
# 	--model "$3" \
# 	--groups 3 \
# 	--version 1.1 \
# 	--width_mult "$4" \
# 	--train_crop random \
# 	--learning_rate 0.001 \
# 	--sample_duration 16 \
# 	--modality RGB \
# 	--downsample 2 \
# 	--batch_size 80 \
# 	--n_threads 16 \
# 	--checkpoint 1 \
# 	--n_val_samples 1 \
#     --n_epochs 70 \
#     --test_subset test \
#     --ft_portion complete 
# 	# --no_train \
#  	# --no_val \
#  	# --test
#  	# --resume_path Efficient-3DCNNs/results/egogesture_shufflenet_2.0x_Depth_16_checkpoint.pth \
#  		# --pretrain_path Efficient-3DCNNs/results/jester_mobilenetv2_1.0x_RGB_16_best.pth \
