python main.py --root_path /home/shared/workspace/Resnet3D/ \
	--video_path 3D-ResNets-PyTorch/data/jpg \
	--annotation_path 3D-ResNets-PyTorch/data/ntu_01.json \
	--result_path Efficient-3DCNNs/data/mobilenetv2_lr1e2_bs32 \
	--resume_path Efficient-3DCNNs/data/mobilenetv2_lr1e2_bs32/ucf101_mobilenetv2_1.0x_RGB_16_checkpoint.pth \
	--dataset ucf101 \
	--n_classes 9 \
	--model mobilenetv2 \
	--groups 3 \
	--width_mult 1.0 \
	--train_crop random \
	--learning_rate 0.01 \
	--sample_duration 16 \
	--downsample 1 \
	--batch_size 16 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
	--no_train \
	--no_val \
	--test