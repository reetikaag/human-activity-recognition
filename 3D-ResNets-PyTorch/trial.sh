python main.py --root_path /home/ubuntu/workspace/Resnet3D/3D-ResNets-PyTorch/ \
	--video_path jpg/new/ \
	--annotation_path annotation/ucf101_01.json \
	--result_path results \
	--dataset ucf101 \
	--model resnet \
	--n_classes 5 \
	--model_depth 50 \
	--batch_size 10 \
	--n_threads 4 \
	--checkpoint 5 \
	--inference \
	--inference_subset val \
	--n_epochs 20 \
	--learning_rate 0.001 \
	--weight_decay 1e-3 \