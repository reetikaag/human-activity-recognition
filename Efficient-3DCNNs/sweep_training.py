import argparse
import os
from subprocess import check_call
import sys
import json
import logging

def launch_training_job(pretrain_path, learning_rate):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    root_path = "/home/shared/workspace/"
    pretrain_name = pretrain_path.split('_')[0]
    model_name = pretrain_path.split('_')[1]
    model_width = pretrain_path.split('_')[2]
    print(pretrain_name)
    print(model_name)
    print(model_width)

    result_name = model_name + "_" + model_width  + "_" + str(50) + "_" + str(learning_rate)
    result_dir = os.path.join(root_path, 'human-activity-recognition', 'Efficient-3DCNNs', 'data', 'results', result_name)
    if not os.path.exists(result_dir):
        print(result_dir)
#         os.makedirs(result_dir)


    # Launch training with this config
    num_classes = 600

    if pretrain_name == "kinetics":
        num_classes = 600
    elif pretrain_name == "jester" :
        num_classes = 27
    print(num_classes)

    cmd = "python main.py --root_path /home/shared/workspace/ \
        --video_path Resnet3D/3D-ResNets-PyTorch/data/jpg \
        --annotation_path Resnet3D/3D-ResNets-PyTorch/data/ntu_01.json \
        --result_path {result_dir} \
        --pretrain_path Resnet3D/3D-ResNets-PyTorch/data/models/{pretrain_path}_RGB_16_best.pth \
        --dataset ucf101 \
        --n_classes {num_classes} \
        --n_finetune_classes 9 \
        --ft_portion last_layer \
        --model {model_name} \
        --groups 3 \
        --width_mult 1.0 \
        --train_crop random \
        --learning_rate {learning_rate} \
        --sample_duration 16 \
        --downsample 1 \
        --n_threads 16 \
        --checkpoint 1 \
        --n_val_samples 1 \
        --batch_size 64 \
        --n_epochs 1".format(result_dir = result_dir, pretrain_path = pretrain_path, num_classes = num_classes,
                             model_name = model_name, learning_rate = learning_rate)
    print(cmd)
    #check_call(cmd, shell=True)

if __name__ == '__main__':
    pretrain_path = ['kinetics_mobilenetv2_1.0x', 'kinetics_shufflenetv2_1.0x', 'kinetics_resnet_101', 'kinetics_resnext_101', 'kinetics_resnet_50', 'kinetics_resnet_18']

    for p in pretrain_path :
        launch_training_job(p, 0.001)
