python main.py --root_path /home/shared/workspace/Resnet3D/3D-ResNets-PyTorch/data/ \
	--video_path jpg \
	--annotation_path ntu_01.json \
	--result_path r3d_101_km_200 \
	--dataset ucf101 \
	--resume_path r3d_101_km_200/save_200.pth \
	--model_depth 101 \
	--n_classes 9 \
	--n_threads 4 \
	--no_train \
	--no_val \
	--inference \
	--output_topk 5 \
	--inference_batch_size 1
