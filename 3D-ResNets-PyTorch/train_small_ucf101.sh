python main.py --root_path /home/ubuntu/workspace/Resnet3D/3D-ResNets-PyTorch/ \
	--video_path jpg/new/ \
	--annotation_path annotation/train_01.json \
	--result_path results \
	--dataset ntu \
	--n_classes 9 \
	--n_pretrain_classes 1039 \
	--pretrain_path models/r3d50_KM_200ep.pth \
	--ft_begin_module fc \
	--model resnet \
	--model_depth 50 \
	--batch_size 10 \
	--n_threads 4 \
	--checkpoint 5 \
	--inference \
	--inference_subset val \
	--n_epochs 200 \
	--learning_rate 0.001 \
	--weight_decay 1e-3 \