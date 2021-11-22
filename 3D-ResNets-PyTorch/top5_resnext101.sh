python main.py --root_path /home/shared/workspace/Resnet3D/3D-ResNets-PyTorch/data/ \
	--video_path jpg \
	--annotation_path ntu_01.json \
	--result_path results/resnext101_K_lr1e2_bs32 \
	--dataset ucf101 \
	--resume_path results/resnext101_K_lr1e2_bs32/save_30.pth \
	--model_depth 101 \
	--model resnext \
	--n_classes 9 \
	--n_threads 4 \
	--no_train \
	--no_val \
	--inference \
	--output_topk 5 \
	--inference_batch_size 1