import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json
import pdb

from utils import AverageMeter, calculate_accuracy


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=2)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })
    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        
        if not opt.no_cuda:
            targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)

        acc = calculate_accuracy(outputs, targets)

        accuracies.update(acc, inputs.size(0))


        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j].item() != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j].item()

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  acc=accuracies))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)
