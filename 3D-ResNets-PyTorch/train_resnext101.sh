python main.py --root_path /home/shared/workspace/Resnet3D/3D-ResNets-PyTorch/data/ \
	--video_path jpg \
	--annotation_path ntu_01.json \
	--result_path results/resnext101_K_lr1e2_bs32 \
	--dataset ucf101 \
	--n_classes 9 \
	--n_pretrain_classes 400 \
	--pretrain_path models/resnext-101-kinetics.pth \
	--ft_begin_module fc \
	--model resnext \
	--resnet_shortcut B \
	--model_depth 101 \
	--batch_size 32 \
	--n_threads 4 \
	--checkpoint 5 \
	--n_epochs 30 \
	--learning_rate 0.01 \
	--weight_decay 1e-3 \


