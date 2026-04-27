import os
import glob
import json
import pandas as pd
import numpy as np
import csv
import torch
import time
from torch.autograd import Variable
from PIL import Image
import cv2
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


###Pretrained RGB models
##Google Drive
#https://drive.google.com/file/d/1V23zvjAKZr7FUOBLpgPZkpHGv8_D-cOs/view?usp=sharing
##Baidu Netdisk
#https://pan.baidu.com/s/114WKw0lxLfWMZA6SYSSJlw code:p1va

def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))


opt = parse_opts_online()

def _select_device(opt):
    """Prefer CUDA, otherwise default to CPU on macOS unless MPS is explicitly requested."""
    if torch.cuda.is_available() and not opt.no_cuda:
        print('Using device: cuda')
        return torch.device('cuda')

    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_mps = os.environ.get('REALTIME_GESREC_USE_MPS', '0') == '1'
    if mps_available and use_mps:
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        print('Using device: mps (fallback enabled for unsupported ops)')
        return torch.device('mps')

    if mps_available:
        print('MPS is available, but using CPU because some 3D ops used by this model are not implemented on MPS.')

    print('Using device: cpu')
    return torch.device('cpu')


device = _select_device(opt)
if device.type != 'cuda':
    opt.no_cuda = True  # tell generate_model not to call .cuda() / DataParallel


def _open_video_capture(video_source):
    """Open a webcam or stream with a macOS-friendly backend when possible."""
    if isinstance(video_source, int) and hasattr(cv2, 'CAP_AVFOUNDATION'):
        cap = cv2.VideoCapture(video_source, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap

    return cv2.VideoCapture(video_source)


def _load_checkpoint(model, path):
    """Load a checkpoint, stripping DataParallel 'module.' prefix when needed."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    if not isinstance(model, torch.nn.DataParallel) and next(iter(state)).startswith('module.'):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)


def load_models(opt):
    opt.resume_path = opt.resume_path_det
    opt.pretrain_path = opt.pretrain_path_det
    opt.sample_duration = opt.sample_duration_det
    opt.model = opt.model_det
    opt.model_depth = opt.model_depth_det
    opt.width_mult = opt.width_mult_det
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
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    detector, parameters = generate_model(opt)
    detector = detector.to(device)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        _load_checkpoint(detector, opt.resume_path)

    print('Model 1 \n', detector)
    pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

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
    classifier = classifier.to(device)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        _load_checkpoint(classifier, opt.resume_path)

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return detector, classifier


detector, classifier = load_models(opt)

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

opt.sample_duration = max(opt.sample_duration_clf, opt.sample_duration_det)

# Load class names from categories.txt if it lives alongside the annotation file
_ann_dir = os.path.dirname(opt.annotation_path) if opt.annotation_path else ''
_cat_file = os.path.join(_ann_dir, 'categories.txt')
class_names = {}
if os.path.exists(_cat_file):
    with open(_cat_file) as _f:
        for _line in _f:
            _parts = _line.strip().split(' ', 1)
            if len(_parts) == 2:
                class_names[int(_parts[0]) - 1] = _parts[1]  # annotations are 1-indexed

fps = ""

# Support both webcam index ("0") and file paths.
_video_src = int(opt.video) if str(opt.video).isdigit() else opt.video
cap = _open_video_capture(_video_src)

if not cap.isOpened():
    raise RuntimeError('Could not open video source: {}. On macOS, try iPhone Continuity Camera, Camo, or an IP camera URL.'.format(opt.video))

"""
# On Linux the FFMPEG backend can't enumerate cameras; force V4L2 for indices.
if str(opt.video).isdigit():
    _video_src = int(opt.video)
    cap = cv2.VideoCapture(_video_src, cv2.CAP_V4L2)
    if not cap.isOpened():  # fall back (e.g. macOS / no V4L2)
        cap = cv2.VideoCapture(_video_src)
else:
    cap = cv2.VideoCapture(opt.video)
"""

num_frame = 0
clip = []
active_index = 0
passive_count = 0
active = False
prev_active = False
finished_prediction = None
pre_predict = False
detector.eval()
classifier.eval()
cum_sum = np.zeros(opt.n_classes_clf, )
clf_selected_queue = np.zeros(opt.n_classes_clf, )
det_selected_queue = np.zeros(opt.n_classes_det, )
myqueue_det = Queue(opt.det_queue_size, n_classes=opt.n_classes_det)
myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)
results = []
prev_best1 = opt.n_classes_clf
spatial_transform.randomize_parameters()
while cap.isOpened():
    t1 = time.time()
    ret, frame = cap.read()
    if not ret or frame is None:
        if num_frame == 0:
            raise RuntimeError('Video source opened but no frames were received from: {}'.format(opt.video))
        break

    if num_frame == 0:
        cur_frame = cv2.resize(frame,(320,240))
        cur_frame = Image.fromarray(cv2.cvtColor(cur_frame,cv2.COLOR_BGR2RGB))
        cur_frame = cur_frame.convert('RGB')
        for i in range(opt.sample_duration):
            clip.append(cur_frame)
        clip = [spatial_transform(img) for img in clip]
    clip.pop(0)
    _frame = cv2.resize(frame,(320,240))
    _frame = Image.fromarray(cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB))
    _frame = _frame.convert('RGB')
    _frame = spatial_transform(_frame)
    clip.append(_frame)
    im_dim = clip[0].size()[-2:]
    try:
        test_data = torch.cat(clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
    except Exception as e:
        pdb.set_trace()
        raise e
    inputs = torch.cat([test_data],0).view(1,3,opt.sample_duration,112,112)
    num_frame += 1


    ground_truth_array = np.zeros(opt.n_classes_clf + 1, )
    with torch.no_grad():
        inputs = inputs.to(device)
        inputs_det = inputs[:, :, -opt.sample_duration_det:, :, :]
        outputs_det = detector(inputs_det)
        outputs_det = F.softmax(outputs_det, dim=1)
        outputs_det = outputs_det.cpu().numpy()[0].reshape(-1, )
        # enqueue the probabilities to the detector queue
        myqueue_det.enqueue(outputs_det.tolist())

        if opt.det_strategy == 'raw':
            det_selected_queue = outputs_det
        elif opt.det_strategy == 'median':
            det_selected_queue = myqueue_det.median
        elif opt.det_strategy == 'ma':
            det_selected_queue = myqueue_det.ma
        elif opt.det_strategy == 'ewma':
            det_selected_queue = myqueue_det.ewma
        prediction_det = np.argmax(det_selected_queue)

        prob_det = det_selected_queue[prediction_det]
        
        #### State of the detector is checked here as detector act as a switch for the classifier
        if prediction_det == 1:
            inputs_clf = inputs.clone()
            outputs_clf = classifier(inputs_clf)
            outputs_clf = F.softmax(outputs_clf, dim=1)
            outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )
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
    
    if passive_count >= opt.det_counter:
        active = False
    else:
        active = True

    # one of the following line need to be commented !!!!
    if active:
        active_index += 1
        cum_sum = ((cum_sum * (active_index - 1)) + (weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
        #cum_sum = ((cum_sum * (active_index-1)) + (1.0 * clf_selected_queue))/active_index #Not Weighting Aproach
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
        #print(finished_prediction,pre_predict)
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
    elapsedTime = time.time() - t1
    fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)

    if len(results) != 0:
        predicted = np.array(results)[:, 1]
        prev_best1 = -1
    else:
        predicted = []

    print('predicted classes: \t', predicted)

    cv2.putText(frame, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
    if results:
        last_cls = int(results[-1][1])
        cv2.putText(frame, class_names.get(last_cls, str(last_cls)),
                    (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", frame)

    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

