python main.py --root_path ~/ \
 	--video_path ~/datasets/Kinetics \
 	--annotation_path Efficient-3DCNNs/annotation_Kinetics/kinetics.json \
 	--result_path Efficient-3DCNNs/results \
 	--dataset kinetics \
 	--n_classes 600 \
 	--sample_size 224 \
 	--model i3d \
 	--train_crop random \
 	--learning_rate 0.1 \
 	--sample_duration 64 \
 	--batch_size 4 \
 	--n_threads 8 \
 	--checkpoint 1 \
 	--n_val_samples 1 \
 #	--no_train \
 #	--no_val \
 #	--test
 # 	--resume_path Efficient-3DCNNs/results/kinetics_resnet_1.0x_RGB_16_checkpoint.pth \




 # python main.py --root_path ~/ \
 # 	--video_path ~/datasets/Kinetics \
 # 	--annotation_path Efficient-3DCNNs/annotation_Kinetics/kinetics.json \
 # 	--result_path Efficient-3DCNNs/results \
 # 	--pretrain_path Efficient-3DCNNs/results/resnet-18-kinetics.pth \
 # 	--resume_path Efficient-3DCNNs/results/kinetics_resnet_1.0x_RGB_16_checkpoint.pth \
 #  --dataset kinetics \
 #  --n_classes 400 \
 #  --n_finetune_classes 600 \
 # 	--sample_size 112 \
 # 	--model resnet \
 # 	--model_depth 18 \
 # 	--resnet_shortcut A \
 # 	--train_crop random \
 # 	--learning_rate 0.1 \
 # 	--sample_duration 16 \
 # 	--batch_size 128 \
 # 	--n_threads 32 \
 # 	--checkpoint 1 \
 # 	--n_val_samples 1 \
 # 	--no_train \
 # 	--no_val \
 # 	--test
 # 	# --resume_path Efficient-3DCNNs/results/kinetics_resnext_1.0x_RGB_16_checkpoint_fnl.pth \