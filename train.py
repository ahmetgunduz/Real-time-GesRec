import torch
from torch.autograd import Variable
import time
import os
import sys
import pdb

from utils import AverageMeter, calculate_accuracy, calculate_precision, calculate_recall


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    precisions = AverageMeter() #
    recalls = AverageMeter()

    end_time = time.time()
    # i, (inputs, targets) = next(iter(enumerate(data_loader)))
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(non_blocking=True)
        inputs = Variable(inputs)
        targets = Variable(targets)
        #pdb.set_trace()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        precision = calculate_precision(outputs, targets) #
        recall = calculate_recall(outputs,targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        precisions.update(precision, inputs.size(0))
        recalls.update(recall,inputs.size(0))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'precision':precisions.val,
            'recall':recalls.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                  'Precision {precision.val:.3f}({precision.avg:.3f})\t'
                  'Recall {recall.val:.3f}({recall.avg:.3f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      lr=optimizer.param_groups[0]['lr'],
                      acc=accuracies,
                      precision=precisions,
                      recall=recalls))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'precision':precisions.avg,
        'recall':recalls.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)
