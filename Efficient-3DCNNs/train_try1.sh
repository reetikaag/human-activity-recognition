python main.py --root_path /home/shared/workspace/Resnet3D/ \
	--video_path 3D-ResNets-PyTorch/data/jpg \
	--annotation_path 3D-ResNets-PyTorch/data/ntu_01.json \
	--result_path Efficient-3DCNNs/data/mobilenetv2_lr1e2_bs32 \
	--pretrain_path 3D-ResNets-PyTorch/data/models/kinetics_mobilenetv2_1.0x_RGB_16_best.pth \
	--dataset ucf101 \
	--n_classes 600 \
	--n_finetune_classes 9 \
	--ft_portion last_layer \
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
