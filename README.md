# Real-time Hand Gesture Recognition with 3D CNNs
PyTorch implementation of the article [Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks](https://arxiv.org/abs/1901.10323), codes and pretrained models.


<div align="center" style="width:image width px;">
  <img  src="https://media.giphy.com/media/9M3aPvPOVxSQmYGv8p/giphy.gif" width=500 alt="simulation results">
</div>

Figure: A real-time simulation of the architecture with input video from EgoGesture dataset (on left side) and real-time (online) classification scores of each gesture (on right side) are shown, where each class is annotated with different color. 


This code includes training, fine-tuning and testing on EgoGesture and nvGesture datasets.  
Note that the code only includes ResNet-10, ResNetL-10, ResneXt-101, C3D v1, whose other versions can be easily added.

## Abstract
Real-time recognition of dynamic hand gestures from video streams is a challenging task since (i) 
there is no indication when a gesture starts and ends in the video, (ii) performed gestures should 
only be recognized once, and (iii) the entire architecture should be designed considering the memory 
and power budget. In this work, we address these challenges by proposing a hierarchical structure 
enabling offline-working convolutional neural network (CNN) architectures to operate online efficiently
by using sliding window approach. The proposed architecture consists of two models: (1) A detector which 
is a lightweight CNN architecture to detect gestures and (2) a classifier which is a deep CNN to classify 
the detected gestures. In order to evaluate the single-time activations of the detected gestures, we propose
to use the Levenshtein distance as an evaluation metric since it can measure misclassifications, multiple detections,
and missing detections at the same time. We evaluate our architecture on two publicly available datasets - EgoGesture
and NVIDIA Dynamic Hand Gesture Datasets - which require temporal detection and classification of the performed hand gestures.
ResNeXt-101 model, which is used as a classifier, achieves the state-of-the-art offline classification accuracy of 94.04% and 
83.82% for depth modality on EgoGesture and NVIDIA benchmarks, respectively. In real-time detection and classification,
we obtain considerable early detections while achieving performances close to offline operation. The codes and pretrained models used in this work are publicly available. 



## Requirements

* [PyTorch](http://pytorch.org/)

```bash
conda install pytorch torchvision cuda80 -c soumith
```

* Python 3

### Pretrained models
[Pretrained_models_v1 (1.08GB)](https://real-time-gesture-models.s3.eu-central-1.amazonaws.com/shared_models_v1.zip): The best performing models in [paper](https://arxiv.org/abs/1901.10323)

[Pretrained_models_v2 (15.2GB)](https://real-time-gesture-models.s3.eu-central-1.amazonaws.com/shared_models_v2.zip): All models in [paper](https://ieeexplore.ieee.org/document/8982092) with efficient 3D-CNN Models
## Preparation

### EgoGesture

* Download videos by following [the official site](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html).
* We will use extracted images that is also provided by the owners

* Generate n_frames files using ```utils/ego_prepare.py``` 

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

```bash
mkdir annotation_EgoGesture
python utils/ego_prepare.py training trainlistall.txt all
python utils/ego_prepare.py training trainlistall_but_None.txt all_but_None
python utils/ego_prepare.py training trainlistbinary.txt binary
python utils/ego_prepare.py validation vallistall.txt all
python utils/ego_prepare.py validation vallistall_but_None.txt all_but_None
python utils/ego_prepare.py validation vallistbinary.txt binary
python utils/ego_prepare.py testing testlistall.txt all
python utils/ego_prepare.py testing testlistall_but_None.txt all_but_None
python utils/ego_prepare.py testing testlistbinary.txt binary
```

* Generate annotation file in json format similar to ActivityNet using ```utils/egogesture_json.py```

```bash
python utils/egogesture_json.py 'annotation_EgoGesture' all
python utils/egogesture_json.py 'annotation_EgoGesture' all_but_None
python utils/egogesture_json.py 'annotation_EgoGesture' binary
```

### nvGesture

* Download videos by following [the official site](https://research.nvidia.com/publication/online-detection-and-classification-dynamic-hand-gestures-recurrent-3d-convolutional).

* Generate n_frames files using ```utils/nv_prepare.py``` 

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

```bash
mkdir annotation_nvGesture
python utils/nv_prepare.py training trainlistall.txt all
python utils/nv_prepare.py training trainlistall_but_None.txt all_but_None
python utils/nv_prepare.py training trainlistbinary.txt binary
python utils/nv_prepare.py validation vallistall.txt all
python utils/nv_prepare.py validation vallistall_but_None.txt all_but_None
python utils/nv_prepare.py validation vallistbinary.txt binary
```

* Generate annotation file in json format similar to ActivityNet using ```utils/nv_json.py```

```bash
python utils/nv_json.py 'annotation_nvGesture' all
python utils/nv_json.py 'annotation_nvGesture' all_but_None
python utils/nv_json.py 'annotation_nvGesture' binary
```

### Jester

* Download videos by following [the official site](https://20bn.com/datasets/jester).

* N frames and class index  file is already provided annotation_Jester/{'classInd.txt', 'trainlist01.txt', 'vallist01.txt'}

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

* Generate annotation file in json format similar to ActivityNet using ```utils/jester_json.py```

```bash
python utils/jester_json.py 'annotation_Jester'
```


## Running the code
* Offline testing (offline_test.py) and training (main.py)
```bash
bash run_offline.sh
```

* Online testing
```bash
bash run_online.sh
```

## Citation

Please cite the following article if you use this code or pre-trained models:

```bibtex
@article{kopuklu_real-time_2019,
	title = {Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks},
	url = {http://arxiv.org/abs/1901.10323},
	author = {Köpüklü, Okan and Gunduz, Ahmet and Kose, Neslihan and Rigoll, Gerhard},
  year={2019}
}
```
## Acknowledgement
We thank Kensho Hara for releasing [3D-ResNets-PyTorch Repo](https://github.com/kenshohara/3D-ResNets-PyTorch), which we build our work on top. 
