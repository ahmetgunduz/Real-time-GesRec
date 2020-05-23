import torch
import torch.utils.data as data
from PIL import Image
from spatial_transforms import *
from temporal_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random

from utils import load_value_file
import pdb


def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Depth':
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, image_loader):
    
    video = []
    if modality == 'RGB':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'Depth':

        for i in frame_indices:
            image_path = os.path.join(video_dir_path.replace('color','depth'), '{:05d}.jpg'.format(i) )
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
            image_path_depth = os.path.join(video_dir_path.replace('color','depth'), '{:05d}.jpg'.format(i) )

            
            image = image_loader(image_path, 'RGB')
            image_depth = image_loader(image_path_depth, 'Depth')

            if os.path.exists(image_path):
                video.append(image)
                video.append(image_depth)
            else:
                print(image_path, "------- Does not exist")
                return video
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_annotation(data, whole_path):
    annotation = []

    for key, value in data['database'].items():
        if key.split('^')[0] == whole_path:
            annotation.append(value['annotations'])

    return  annotation


def make_dataset( annotation_path, video_path , whole_path,sample_duration, n_samples_for_each_video, stride_len):
    
    data = load_annotation_data(annotation_path)
    whole_video_path = os.path.join(video_path,whole_path)
    annotation = get_annotation(data, whole_path)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print("[INFO]: Videot  is loading...")
    import glob

    n_frames = len(glob.glob(whole_video_path + '/*.jpg'))
    
    if not os.path.exists(whole_video_path):
        print(whole_video_path , " does not exist")
    label_list = []
    for i in range(len(annotation)):
        begin_t = int(annotation[i]['start_frame'])
        end_t = int(annotation[i]['end_frame'])
        for j in range(begin_t,end_t+1):
            label_list.append(class_to_idx[annotation[i]['label']])

    label_list = np.array(label_list)
    for _ in range(1,n_frames+1 - sample_duration,stride_len):
        
        sample = {
                'video': whole_video_path,
                'index': _ ,
                'video_id' : _ 

            }
        ## Different strategies to set true label of overlaping frames
        # counts = np.bincount(label_list[np.array(list(range(_    - int(sample_duration/4), _ )))])
        sample['label'] = 0 #np.argmax(counts)
        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(_ , _ + sample_duration))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                            math.ceil((n_frames - 1 - sample_duration) /
                                    (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(sample_duration, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class




class NVOnline(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 annotation_path,
                 video_path,
                 whole_path,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 stride_len = None,
                 get_loader=get_default_video_loader):

        self.data, self.class_names = make_dataset(
         annotation_path, video_path, whole_path, sample_duration,n_samples_for_each_video, stride_len)
        
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)
        oversample_clip =[]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return clip, target

    def __len__(self):
        return len(self.data)


