from pathlib import Path
import json
import random
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
import csv
import matplotlib.pyplot as plt
import cv2

ACTION_NAMES = ["sneezeCough", "staggering", "fallingDown",
                "headache", "chestPain", "backPain",
                "neckPain", "nauseaVomiting", "fanSelf"]


def get_saliency_map(opt, model, X, y):
    """
    This is a function added for 231n.
    Compute a class saliency map using the model for single video X and label y.
    Input:
    - X: Input video; Tensor of shape (1, 3, T, H, W) -- 1 video, 3 channels, T frames, HxW images
    - y: Labels for X; LongTensor of shape (1,) -- 1 label
    - model: A pretrained CNN that will be used to compute the saliency map.
    Returns:
    - saliency: A Tensor of shape (1, T, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensors require gradient
    X.requires_grad_()
    saliency = None
    # Convert y (targets) into labels
    labels = []
    for elem in y:
        left, _ = elem.split("_")
        label = int(left[-2:]) - 41
        labels.append(label)

    y = torch.LongTensor(labels)
    if not opt.no_cuda:
        y = y.cuda()
    scores = model(X).gather(1, y.view(-1, 1)).squeeze().sum()
    scores.backward()
    saliency, temp = X.grad.data.abs().max(dim = 1)
    return saliency


def plot_saliency(sal_map, i, inputs, targets):
    # Use matplotlib to make one figure showing the average image for each segment
    # for the video and the saliency map for each segment of the video

    # For a video with 5 segments which results in sal_map 5x16x112x112
    # We avg over the 16 saliency maps (one for each image in the segment) to get 5x112x112
    # inputs has shape 5x3x16x112x112 --> this is the segment of input images
    # Avg over 16 images in the segment and take max over 3 channels of each image
    # Plot each of the 5 images with corresponding heatmap of saliency
   
    with torch.no_grad():
        sal_map = sal_map.numpy()
        inputs = inputs.detach().numpy()
        # 1. Average over saliency map dimensions
        avg_sal_map = np.mean(sal_map[:,3:,:,:], axis=1)

        # 2. Average over image dimensions
        avg_inputs = np.mean(inputs[:,:,3:,:,:], axis=2)
        max_inputs = np.mean(avg_inputs, axis=1)
        # 3. Convert targets into labels
        labels = []
        for elem in targets:
            label = int(elem.split('_')[0][-2:]) - 41
            labels.append(label)
        y = torch.LongTensor(labels)
        # 3. Make a plt figure and put the images in their correct positions and save to file
        N = sal_map.shape[0]
        for j in range(N):
            fig = plt.figure(figsize=(9,9))
            ax = fig.add_subplot(2, N, j + 1)
            ax.imshow(max_inputs[j])
            #plt.imshow(max_inputs[j])
            ax.axis('off')
            fig.suptitle(ACTION_NAMES[y[j]])
            ax2 = fig.add_subplot(2, N, N + j + 1)
            ax2.imshow(avg_sal_map[j], cmap=plt.cm.hot)
            #plt.imshow(avg_sal_map[j], cmap=plt.cm.hot)
            ax2.axis('off')
            #plt.gcf().set_size_inches(12, )
            plt.show()
        figpath = Path('/home/shared/workspace/human-activity-recognition/Efficient-3DCNNs/data/results/saliency_maps/average'+ACTION_NAMES[y[j]]+str(i))
        fig.savefig(figpath)
        center_frame = int(avg_inputs.shape[1]/2)
        #for j in range(N):
        #    plt.subplot(2, N, j + 1)
        #    plt.imshow(avg_inputs[j,center_frame,:,:])
        #    plt.axis('off')
        #    plt.title(ACTION_NAMES[y[j]])
        #    plt.subplot(2, N, N + j + 1)
        #    plt.imshow(sal_map[j,center_frame,:,:], cmap=plt.cm.hot)
        #    plt.axis('off')
            #plt.gcf().set_size_inches(12, )
        #    plt.show()
        #figpath = Path('/home/shared/workspace/human-activity-recognition/Efficient-3DCNNs/data/results/saliency_maps/center'+ACTION_NAMES[y[j]]+str(i))
        #plt.savefig(figpath)
    return None
