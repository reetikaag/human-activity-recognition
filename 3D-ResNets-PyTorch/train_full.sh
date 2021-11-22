python main.py --root_path /home/shared/workspace/Resnet3D/3D-ResNets-PyTorch/data/ \
	--video_path jpg \
	--annotation_path ntu_01.json \
	--result_path r3d_101_km_200_lr_1e4 \
	--dataset ucf101 \
	--n_classes 9 \
	--n_pretrain_classes 1039 \
	--pretrain_path models/r3d101_KM_200ep.pth \
	--ft_begin_module fc \
	--model resnet \
	--model_depth 101 \
	--batch_size 32 \
	--n_threads 4 \
	--checkpoint 5 \
	--n_epochs 30 \
	--learning_rate 0.0001 \
	--weight_decay 1e-3 \


