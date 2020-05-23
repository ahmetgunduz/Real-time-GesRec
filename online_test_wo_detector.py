import os
import glob
import json
import pandas as pd
import csv
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel
from dataset import get_online_data
from utils import  AverageMeter, LevenshteinDistance, Queue

import pdb
import numpy as np
import datetime


def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))


opt = parse_opts_online()


opt = parse_opts_online()


def load_models(opt):
    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.width_mult = opt.width_mult_clf
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
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
#        assert opt.arch == checkpoint['arch']

        classifier.load_state_dict(checkpoint['state_dict'])

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return classifier


classifier = load_models(opt)


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

## Get list of videos to test
if opt.dataset == 'egogesture':
    subject_list = ['Subject{:02d}'.format(i) for i in [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]]
    test_paths = []
    for subject in subject_list:
        for x in glob.glob(os.path.join(opt.video_path, subject, '*/*/rgb*')):
            test_paths.append(x)
elif opt.dataset == 'nvgesture':
    df = pd.read_csv(os.path.join(opt.video_path, 'nvgesture_test_correct_cvpr2016_v2.lst'), delimiter=' ', header=None)
    test_paths = []
    for x in df[0].values:
        test_paths.append(os.path.join(opt.video_path, x.replace('path:', ''), 'sk_color_all'))

print('Start Evaluation')
classifier.eval()


levenshtein_accuracies = AverageMeter()
videoidx = 0
for path in test_paths[:]:
    if opt.dataset == 'egogesture':
        opt.whole_path = os.path.join(*path.rsplit(os.sep, 4)[1:])
    elif opt.dataset == 'nvgesture':
        opt.whole_path = os.path.join(*path.rsplit(os.sep, 5)[1:])

    videoidx += 1
    active_index = 0
    passive_count = 0
    active = False
    prev_active = False
    finished_prediction = None
    pre_predict = False

    cum_sum = np.zeros(opt.n_classes_clf, )
    clf_selected_queue = np.zeros(opt.n_classes_clf, )
    det_selected_queue = np.zeros(opt.n_classes_det, )
    myqueue_det = Queue(opt.det_queue_size, n_classes=opt.n_classes_det)
    myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)

    print('[{}/{}]============'.format(videoidx, len(test_paths)))
    print(path)
    opt.sample_duration = max(opt.sample_duration_clf, opt.sample_duration_det)
    temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
    test_data = get_online_data(
        opt, spatial_transform, None, target_transform)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)

    results = []
    prev_best1 = opt.n_classes_clf
    dataset_len = len(test_loader.dataset)
    for i, (inputs, targets) in enumerate(test_loader):
        if not opt.no_cuda:
            targets = targets.cuda()
        ground_truth_array = np.zeros(opt.n_classes_clf + 1, )
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
            if opt.modality_clf == 'RGB':
                inputs_clf = inputs[:, :-1, :, :, :]
            elif opt.modality_clf == 'Depth':
                inputs_clf = inputs[:, -1, :, :, :].unsqueeze(1)
            elif opt.modality_clf == 'RGB-D':
                inputs_clf = inputs[:, :, :, :, :]
            inputs_clf = torch.Tensor(inputs_clf.numpy()[:, :, ::2, :, :])
            outputs_clf = classifier(inputs_clf)
            outputs_clf = F.softmax(outputs_clf, dim=1)
            outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )

            #pdb.set_trace()
            #### State of the detector is checked here as detector act as a switch for the classifier
            if  np.argmax(outputs_clf) != opt.n_classes_clf-1:
                # Push the probabilities to queue
                myqueue_clf.enqueue(outputs_clf.tolist())
                passive_count = 0

                if opt.clf_strategy == 'raw':
                    clf_selected_queue = outputs_clf
                elif opt.clf_strategy == 'median':
                    clf_selected_queue = myqueue_clf.median
                elif opt.clf_strategy == 'ma':
                    clf_selected_queue = myqueue_clf.ma
                elif opt.clf_strategy == 'ewma':
                    clf_selected_queue = myqueue_clf.ewma

            else:
                outputs_clf = np.zeros(opt.n_classes_clf, )
                # Push the probabilities to queue
                myqueue_clf.enqueue(outputs_clf.tolist())
                passive_count += 1

        if passive_count >= opt.det_counter or i == (dataset_len -2):
            active = False
        else:
            active = True

        # one of the following line need to be commented !!!!
        if active:
            active_index += 1
            cum_sum = ((cum_sum * (active_index - 1)) + (
                        weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
            # cum_sum = ((cum_sum * (x-1)) + (1.0 * clf_selected_queue))/x #Not Weighting Aproach

            best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
            if float(cum_sum[best1] - cum_sum[best2]) > opt.clf_threshold_pre:
                finished_prediction = True
                pre_predict = True

        else:
            active_index = 0

        if active == False and prev_active == True:
            finished_prediction = True
        elif active == True and prev_active == False:
            finished_prediction = False

        if finished_prediction == True:
            best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
            if cum_sum[best1] > opt.clf_threshold_final:
                if pre_predict == True:
                    if best1 != prev_best1:
                        if cum_sum[best1] > opt.clf_threshold_final:
                            results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                            print('Early Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
                                                                                                  (
                                                                                                              i * opt.stride_len) + opt.sample_duration_clf))
                else:
                    if cum_sum[best1] > opt.clf_threshold_final:
                        if best1 == prev_best1:
                            if cum_sum[best1] > 5:
                                results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                                print('Late Detected - class : {} with prob : {} at frame {}'.format(best1,
                                                                                                     cum_sum[best1], (
                                                                                                                 i * opt.stride_len) + opt.sample_duration_clf))
                        else:
                            results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))

                            print('Late Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
                                                                                                 (
                                                                                                             i * opt.stride_len) + opt.sample_duration_clf))

                finished_prediction = False
                prev_best1 = best1

            cum_sum = np.zeros(opt.n_classes_clf, )

        if active == False and prev_active == True:
            pre_predict = False

        prev_active = active


    if opt.dataset == 'egogesture':
        target_csv_path = os.path.join(opt.video_path,
                                       'labels-final-revised1',
                                       opt.whole_path.rsplit(os.sep, 2)[0],
                                       'Group' + opt.whole_path[-1] + '.csv').replace('Subject', 'subject')
        true_classes = []
        with open(target_csv_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                true_classes.append(int(row[0]) - 1)
    elif opt.dataset == 'nvgesture':
        true_classes = []
        with open('./annotation_nvGesture/vallistall.txt') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=' ')
            for row in readCSV:
                if row[0] == opt.whole_path:
                    if row[1] != '26':
                        true_classes.append(int(row[1]) - 1)
    if len(results) != 0:
        predicted = np.array(results)[:, 1]
    else:
        predicted = []
    true_classes = np.array(true_classes)
    levenshtein_distance = LevenshteinDistance(true_classes, predicted)
    levenshtein_accuracy = 1 - (levenshtein_distance / len(true_classes))
    if levenshtein_distance < 0:  # Distance cannot be less than 0
        levenshtein_accuracies.update(0, len(true_classes))
    else:
        levenshtein_accuracies.update(levenshtein_accuracy, len(true_classes))

    print('predicted classes: \t', predicted)
    print('True classes :\t\t', true_classes)
    print('Levenshtein Accuracy = {} ({})'.format(levenshtein_accuracies.val, levenshtein_accuracies.avg))

print('Average Levenshtein Accuracy= {}'.format(levenshtein_accuracies.avg))

print('-----Evaluation is finished------')
with open("./results/online-results.log", "a") as myfile:
    myfile.write("{}, {}, {}, {}, {}, {}".format(datetime.datetime.now(),
                                    opt.resume_path_clf,
                                    opt.model_clf,
                                    opt.width_mult_clf,
                                    opt.modality_clf,
                                    levenshtein_accuracies.avg))
