import argparse
import time
import os
import glob 
import sys
import json
import shutil
import itertools
import numpy as np
import pandas as pd 
import csv
import torch
import tensorflow as tf
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_video_data ,get_training_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
from utils import AverageMeter, calculate_precision, calculate_recall
import pdb
import ctcdecode

import matplotlib.pyplot as plt
import matplotlib.animation as animation 
from matplotlib.pyplot import figure
# figure(num=None, figsize=(9, 3), dpi=180, facecolor='w', edgecolor='k')

import cv2
import PIL 
from PIL import ImageFont, ImageDraw, Image

# Reshape a numpy array 'a' of shape (n, x) to form shape((n - window_size), window_size, x))
def rolling_window(a, window, step_size):
    a = a.transpose()
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def sigmoidlike(x):
    return (1 / (1 + np.exp(-0.2*(x-9))))






def levenshtein(a,b):
    # This is a straightforward implementation of a well-known algorithm, and thus
    # probably shouldn't be covered by copyright to begin with. But in case it is,
    # the author (Magnus Lie Hetland) has, to the extent possible under law,
    # dedicated all copyright and related and neighboring rights to this software
    # to the public domain worldwide, by distributing it under the CC0 license,
    # version 1.0. This software is distributed without any warranty. For more
    # information, see <http://creativecommons.org/publicdomain/zero/1.0>
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]



class Queue:
    #Constructor creates a list
    def __init__(self, max_size, n_classes):
        self.queue = list(np.zeros((max_size, n_classes),dtype = float).tolist())
        self.max_size = max_size
    #Adding elements to queue
    def enqueue(self,data):
        self.queue.insert(0,data)
        return True

    #Removing the last element from the queue
    def dequeue(self):
        if len(self.queue)>0:
            return self.queue.pop()
        return ("Queue Empty!")

    #Getting the size of the queue
    def size(self):
        return len(self.queue)

    #printing the elements of the queue
    def printQueue(self):
        return self.queue

    #Average   
    def ma(self):
        return np.array(self.queue[:self.max_size]).mean(axis = 0)

    #Median
    def median(self):
        return np.median(np.array(self.queue[:self.max_size]), axis = 0)
    
    #Exponential average
    def ewma(self):
        weights = np.exp(np.linspace(-1., 0., self.max_size))
        weights /= weights.sum()
        average = weights.reshape(1,self.max_size).dot( np.array(self.queue[:self.max_size]))
        return average.reshape(average.shape[1],)


def calculate_accuracy(outputs, targets, topk=(1,)):
    # Helper function to calculate top k accuracy (top1 in binary case)
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    ret = []
    for k in topk:
        correct_k = correct[:k].float().sum().item()
        ret.append(correct_k / batch_size)

    return ret


opt = parse_opts_online()

def load_models(opt):
    opt.resume_path = opt.resume_path_det
    opt.pretrain_path = opt.pretrain_path_det
    opt.sample_duration = opt.sample_duration_det
    opt.model = opt.model_det
    opt.model_depth = opt.model_depth_det
    opt.modality = opt.modality_det
    opt.resnet_shortcut = opt.resnet_shortcut_det
    opt.n_classes = opt.n_classes_det
    opt.n_finetune_classes = opt.n_finetune_classes_det

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)




    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    detector, parameters = generate_model(opt)

    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        detector.load_state_dict(checkpoint['state_dict'])

    print('Detector \n', detector)
    pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)


    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        classifier.load_state_dict(checkpoint['state_dict'])

    print('Classifier \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return detector, classifier

detector,classifier = load_models(opt)

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)


spatial_transform = Compose([
    Scale(112),
    CenterCrop(112),
    ToTensor(opt.norm_value), norm_method
    ])

target_transform = ClassLabel()

print('run')

det_strategy_list = ['median']
det_queue_size_list = [4]
det_threshold_list = [2]
clf_strategy_list = [ 'median']
clf_queue_size_list = [16]
clf_threshold_pre_list = [ 0.9, 1.0]
clf_threshold_final_list = [0.15]

combinations_list = [det_strategy_list,\
                    det_queue_size_list,\
                    det_threshold_list,\
                    clf_strategy_list,\
                    clf_queue_size_list, \
                    clf_threshold_pre_list, \
                    clf_threshold_final_list]
combinations_list =  list(itertools.product(*combinations_list))

## To get list of videos
if opt.dataset == 'egogesture':
    subject_list = ['Subject{:02d}'.format(i) for i in [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]]
    test_paths = []
    for subject in subject_list:
        for x in glob.glob(os.path.join(opt.video_path,subject,'*/*/rgb*')):
            test_paths.append(x)
elif opt.dataset == 'nv':
    df = pd.read_csv(os.path.join(opt.video_path,'nvgesture_test_correct_cvpr2016_v2.lst'), delimiter = ' ', header = None)
    test_paths = []
    for x in df[0].values:
        test_paths.append(os.path.join(opt.video_path, x.replace('path:', ''), 'sk_color_all'))


for comb in combinations_list:

    opt.det_strategy = comb[0]
    opt.det_queue_size = comb[1]
    opt.det_threshold = comb[2]
    opt.clf_strategy = comb[3]
    opt.clf_queue_size = comb[4]
    opt.clf_threshold_pre = comb[5]
    opt.clf_threshold_final = comb[6]

    detector.eval()
    classifier.eval()

    levenstein_accuracies = AverageMeter()
    videoidx = 0
    for path in test_paths[4:]:
        if opt.dataset == 'egogesture':
            opt.whole_path = path.split(os.sep, 4)[-1]
        elif opt.dataset == 'nv':
            opt.whole_path = path.split(os.sep, 3)[-1]
        
        new_row = []
        new_row_result = []
        videoidx += 1
        # Initialize the buffer for the logits
        recorderma_det     = []
        recorderewma_det   = []
        recorderraw_det    = []
        recordermedian_det = []
        recorderma_clf     = []
        recorderewma_clf   = []
        recorderraw_clf    = []
        recordermedian_clf = []
        recordercumsum = []
        recordermaplot  = [] 
        recordermedianplot  = []
        recorderewmaplot  = []
        recorder_ground_truth = []
        recorder_state  = [] 
        recorderindexcumsum = []


        x = 0
        active_count = 0
        passive_count = 0
        active = False
        prev_active = False
        finished_prediction = None
        started_prediction = None
        pre_predict = False
        cum_sum = np.zeros(opt.n_classes_clf,)
        clf_selected_queue = np.zeros(opt.n_classes_clf,)
        det_selected_queue = np.zeros(opt.n_classes_det,)
        myqueue_det = Queue(opt.det_queue_size ,  n_classes = opt.n_classes_det)
        myqueue_clf = Queue(opt.clf_queue_size, n_classes = opt.n_classes_clf )


        print('[{}/{}]----------------'.format(videoidx,len(test_paths)))
        print(path)
        test_data = get_video_data(
            opt, spatial_transform, None, target_transform)

        test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=True)


        results = []
        prev_best1 = opt.n_classes_clf

        new_row.append(path)
        start_frame = int(16)
        new_row.append(start_frame)
        max_i  = len(list(enumerate(test_loader))) -1
        for i, (inputs, targets) in enumerate(test_loader):
            if (i %100) == 0 :
                print(i)
            if not opt.no_cuda:
                targets = targets.cuda(async=True)
            ground_truth_array = np.zeros(opt.n_classes_clf +1,)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                if opt.modality1 == 'RGB':
                    inputs_det = inputs[:,:-1,-opt.sample_duration_det:,:,:]
                elif opt.modality1 == 'Depth':
                    inputs_det = inputs[:,-1,-opt.sample_duration_det:,:,:].unsqueeze(1)
                elif opt.modality1 =='RGB-D':
                    inputs_det = inputs[:,:,-opt.sample_duration_det:,:,:]
                
                outputs_det = detector(inputs_det)
                outputs_det = F.softmax(outputs_det,dim=1)
                outputs_det = outputs_det.cpu().numpy()[0].reshape(-1,)

                # enqueue the probabilities to the detector queue
                myqueue_det.enqueue(outputs_det.tolist())

                # Calculate moving averages
                moving_average_det = myqueue_det.ma()
                emoving_average_det = myqueue_det.ewma()
                moving_median_det = myqueue_det.median()

                if opt.det_strategy == 'raw':
                    det_selected_queue = outputs_det
                elif opt.det_strategy == 'median':
                    det_selected_queue = moving_median_det
                elif opt.det_strategy == 'ma':
                    det_selected_queue = moving_average_det
                elif opt.det_strategy == 'ewma':
                    det_selected_queue = emoving_average_det
                

                prediction_det = np.argmax(det_selected_queue)
                prob1 = det_selected_queue[prediction_det]
                
                if  prediction_det == 1:
                    if opt.modality_clf == 'RGB':
                        inputs_clf = inputs[:,:-1,:,:,:]
                    elif opt.modality_clf == 'Depth':
                        inputs_clf = inputs[:,-1,:,:,:].unsqueeze(1)
                    elif opt.modality_clf =='RGB-D':
                        inputs_clf = inputs[:,:,:,:,:]

                    outputs_clf = classifier(inputs_clf)
                    outputs_clf = F.softmax(outputs_clf,dim=1)
                    outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1,)
                    

                    # Push the probabilities to queue
                    myqueue_clf.enqueue(outputs_clf.tolist())

                    # Calcualte moving averages
                    moving_average_clf = myqueue_clf.ma()
                    emoving_average_clf = myqueue_clf.ewma()
                    moving_median_clf = myqueue_clf.median()

                    recordermaplot.append(moving_average_clf) 
                    recordermedianplot.append(moving_median_clf)
                    recorderewmaplot.append(emoving_average_clf)
                    passive_count = 0
                    active_count += 1

                    if opt.clf_strategy == 'raw':
                        clf_selected_queue = outputs_clf
                    elif opt.clf_strategy == 'median':
                        clf_selected_queue = moving_median_clf
                    elif opt.clf_strategy == 'ma':
                        clf_selected_queue = moving_average_clf
                    elif opt.clf_strategy == 'ewma':
                        clf_selected_queue = emoving_average_clf


                    best1 = np.argmax(clf_selected_queue)
                    prob = clf_selected_queue[best1]

                else:
                    outputs_clf = np.zeros(opt.n_classes_clf ,)
                    # Push the probabilities to queue
                    myqueue_clf.enqueue(outputs_clf.tolist())
                    # Calcualte moving averages
                    moving_average_clf = myqueue_clf.ma()
                    emoving_average_clf = myqueue_clf.ewma()
                    moving_median_clf = myqueue_clf.median()

                    prob = prob1
                    best1 = opt.n_classes_clf
                    recordermaplot.append(np.zeros(opt.n_classes_clf ,)) 
                    recordermedianplot.append(np.zeros(opt.n_classes_clf ,))
                    recorderewmaplot.append(np.zeros(opt.n_classes_clf ,))

                    passive_count += 1
                    active_count = 0
            

            #pdb.set_trace()
            if passive_count >= opt.det_threshold  or i == max_i:
                active = False
            else:
                active = True



            if active:
                recorder_state.append(1)
                x += 1
                cum_sum = ((cum_sum * (x-1)) + (sigmoidlike(x) * clf_selected_queue))/x
                #cum_sum = ((cum_sum * (x-1)) + (1.0 * clf_selected_queue))/x

                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if float(cum_sum[best1]- cum_sum[best2]) > opt.clf_threshold_pre:
                    finished_prediction = True
                    pre_predict = True
                
            else:
                x = 0


            if active == False and  prev_active == True:
                finished_prediction = True
                started_prediction = False
            elif active == True and  prev_active == False:
                started_prediction = True
                finished_prediction = False



            if finished_prediction == True:
                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                
                
                if cum_sum[best1]>opt.clf_threshold_final:
                    if pre_predict == True:  
                        if best1 != prev_best1:
                            if cum_sum[best1]>opt.clf_threshold_final:  
                                results.append(((i*opt.stride_len)+opt.sample_duration2,best1))
                                print( '1 -- candidate : {} , prob : {} at frame {}'.format(best1, cum_sum[best1], (i*opt.stride_len)+opt.sample_duration2))                      
                    else:
                        if cum_sum[best1]>opt.clf_threshold_final:
                            if best1 == prev_best1:
                                if cum_sum[best1]>5:
                                    results.append(((i*opt.stride_len)+opt.sample_duration2,best1))
                                    print( '2 --candidate : {} , prob : {} at frame {}'.format(best1, cum_sum[best1], (i*opt.stride_len)+opt.sample_duration2))
                            else:
                                results.append(((i*opt.stride_len)+opt.sample_duration2,best1))
                                
                                print( '2 --candidate : {} , prob : {} at frame {}'.format(best1, cum_sum[best1], (i*opt.stride_len)+opt.sample_duration2))

                            

                    finished_prediction = False
                    prev_best1 = best1

                cum_sum = np.zeros(opt.n_classes_clf,)

            if active == False and  prev_active == True:
                pre_predict = False
        
            prev_active = active

            ground_truth_array[targets.item()] = 1.0
            recorder_ground_truth.append(ground_truth_array)
            # pdb.set_trace()
            # Append moving averages
            recorderma_det.append(moving_average_det)
            recorderewma_det.append(emoving_average_det)
            recorderraw_det.append(outputs_det)
            recordermedian_det.append(moving_median_det)
            recordercumsum.append(cum_sum)
            recorderindexcumsum.append(x)
            recorderma_clf.append(moving_average_clf)
            recorderewma_clf.append(emoving_average_clf)
            recorderraw_clf.append(outputs_clf)
            recordermedian_clf.append(moving_median_clf) 
            
        

        ## Print outputs for the video
        recordermedianplot = np.array(recordermedianplot)
        recorderewmaplot = np.array(recorderewmaplot)
        recordermaplot = np.array(recordermaplot)
        recorderraw_det = np.array(recorderraw_det) 
        recordermedian_det = np.array(recordermedian_det) 
        recorderma_det = np.array(recorderma_det)     
        recorderraw_clf = np.array(recorderraw_clf)
        recorderma_clf = np.array(recorderma_clf)
        recordercumsum = np.array(recordercumsum)
        recorderewma_clf = np.array(recorderewma_clf)
        recordermedian_clf = np.array(recordermedian_clf)
        recorder_ground_truth = np.array(recorder_ground_truth)
        recorderall = np.concatenate([recorderraw_clf,  recorderraw_det[:,1].reshape(recorderraw_det.shape[0],1)], axis = 1)

        if opt.dataset == 'egogesture':
            target_csv_path = os.path.join(opt.video_path.rsplit(os.sep, 1)[0], 
                                    'labels-final-revised1',
                                    opt.whole_path.rsplit(os.sep,2)[0],
                                    'Group'+opt.whole_path[-1] + '.csv').replace('Subject', 'subject')
            target_list = []
            with open(target_csv_path) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    target_list.append(int(row[0])-1)
        elif opt.dataset == 'nv':
            target_list = []
            with open('./annotation_nvGesture/vallistall.txt') as csvfile:
                readCSV = csv.reader(csvfile, delimiter=' ')
                for row in readCSV:
                    if row[0] == opt.whole_path:
                        if row[1] != '26' :
                            target_list.append(int(row[1])-1)
        try:
            result_list = np.array(results)[:,1]
        except:
            result_list = np.array([])
        
        target_list = np.array(target_list)
        distance = levenshtein(target_list, result_list)
        
        if (1-(distance/len(target_list))) <0:
            levenstein_accuracies.update(0, len(target_list))
        else:
            levenstein_accuracies.update(1-(distance/len(target_list)), len(target_list))

        
        print('results :',result_list)
        print('targets :',target_list)
        print('Accuracy = {} ({})'.format(levenstein_accuracies.val, levenstein_accuracies.avg))
        plt.plot(recorderraw_clf)
        plt.show()


        pdb.set_trace()




print('-----Evaluation is finished------')
        # print('Overall Prec@1 {:.05f}% Prec@5 {:.05f}%'.format(top1.avg, top5.

    # plt.plot(recorderewma_clf)
    # plt.show()
    # plt.plot( pd.ewma(recorderraw_clf,com = 0.5,  min_periods = 1)) # Exponential weigted moving average
    # plt.plot( pd.rolling_mean(recorderraw_clf, 5, min_periods = 1)) # Moving average 
