import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json

from utils import AverageMeter
from saliency import get_saliency_map, plot_saliency

def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    #print("video_id is {} {}".format(video_id, video_outputs.shape[0]))
    iter = video_outputs.shape[0]
    mid = int(iter/2)
    #average_scores = torch.mean(video_outputs[2:-1,:], dim=0)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=3)
    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[int(locs[i])],
            'score': float(sorted_scores[i])
        })
    test_results['results'][video_id] = video_results
    #print(video_id, video_results)

def calculate_midframe_accuracy(output_buffer, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = 1 
    video_outputs = torch.stack(output_buffer)
    iter = video_outputs.shape[0]
    mid = int(iter/2)
    output = torch.mean(video_outputs[mid:mid+1,:], dim=0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        #print(i, targets)
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            inputs = Variable(inputs)
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                #res = calculate_midframe_accuracy(output_buffer, previous_video_id)
                #print(res)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]
        
        #saliency_maps = []
        #sal_map = get_saliency_map(opt, model, inputs, targets)
        #plot_saliency(sal_map, i, inputs, targets)
        #saliency_maps.append(sal_map)

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    
    if i == len(data_loader) - 1:
        calculate_video_results(output_buffer, previous_video_id, test_results, class_names)
        #sal_map = get_saliency_map(opt, model, inputs, targets)
        #plot_saliency(sal_map, i, inputs, targets)
        #saliency_maps.append(sal_map)
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)
