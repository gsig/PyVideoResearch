## PyVideoResearch

* A repositsory of common methods, datasets, and tasks for video research

* Please note that this repository is in the process of being released to the public. Please bear with us as we standardize the API and streamline the code. 
* Some of the baselines were run with an older version of the codebase (but the git commit hash is available for each experiment) and might need to be updated. 
* We encourage you to submit a Pull Request to help us document and incorporate as many baselines and datasets as possible to this codebase
* We hope this project will be of value to the community and everyone will consider adding their methods to this codebase

List of implemented methods:
* I3D
* 3D ResNet
* Asynchronous Temporal Fields
* Actor Observer Network
* Temporal Segment Networks
* Temporal Relational Networks
* Non-local neural networks
* Two-Stream Networks
* I3D Mask-RCNN 
* 3D ResNet Video Autoencoder

List of supported datasets:
* Charades
* CharadesEgo
* Kinetics
* AVA
* ActivityNet
* Something Something
* Jester

List of supported tasks:
* Action classification
* Action localization
* Spatial Action localization
* Inpainting
* Video Alignment
* Triplet Classification

Contributor: Gunnar Atli Sigurdsson

* If this code helps your research, please consider citing: 

```
@inproceedings{sigurdsson2018pyvideoresearch,
author = {Gunnar A. Sigurdsson and Abhinav Gupta},
title = {PyVideoResearch},
year={2018},
code = {https://github.com/gsig/PyVideoResearch},
}
```
and remember to cite the papers for the datasets/methods you use.

## Installation Instructions

Requirements:
* Python 2.7 or Python 3.6
* PyTorch 0.4 or PyTorch 1.0

Python packages:
* numpy
* ffmpeg-python
* PIL
* cv2
* torchvision

See external libraries under external/ for requirements if using their corresponding baselines. 

Run the following to get both this repository and the remote repositories under external/

```
git clone git@github.com:gsig/PyVideoResearch.git
git submodule update --init --recursive
```


## Steps to train your own network:
 
1. Download the corresponding dataset 
2. Duplicate and edit one of the experiment files under exp/ with appropriate parameters. For additional parameters, see opts.py
3. Run an experiment by calling python exp/rgbnet.py where rgbnet.py is your experiment file. See baseline_exp/ for a variety of baselines.
4. The checkpoints/logfiles/outputs are stored in your specified cache directory. 
5. Build of the code, cite our papers, and say hi to us at CVPR.

Good luck!


## Pretrained networks:

We are in the process of preparing and releasing the pre-trained models. If anything is missing, please let us know. The names correspond to experiments under "baseline_exp". While we standardize the names, please be aware that some of the model may have names listed after "original name" in the experiment file. We also provide the generated log.txt file for each experiment as name.txt

The models are stored here: https://www.dropbox.com/sh/duodxydolzz5qfl/AAC0i70lv8ssVRWg4ux5Vv9pa?dl=0

* ResNet50 pre-trained on Charades
    * resnet50_rgb.pth.tar
    * resnet50_rgb_python3.pth.tar
* ResNet1010 pre-trained on Charades
    * resnet101_rgb.pth.tar
    * resnet101_rgb_python3.pth.tar
* I3D pre-trained on ImageNet+Kinetics (courtesy of https://github.com/piergiaj)
    * aj_rgb_imagenet.pth
* I3D pre-trained on Charades (courtesy of https://github.com/piergiaj)
    * aj_rgb_charades.pth

* actor_observer_3d_charades_ego.py
* actor_observer_charades_ego.py
* actor_observer_classification_charades_ego.py
* async_tf_i3d_charades.py
    * async__par1.pth.tar
    * async__par1.txt
* i3d_ava.py
* i3d_mask_rcnn_ava.py
* i3d_something_something.py
* inpainting.py
* nonlocal_resnet50_3d_charades.py
    * i3d31b.pth.tar
    * i3d31b.pth.tar
* nonlocal_resnet50_3d_kinetics.py
    * i3d8l.pth.tar
    * i3d8l.txt
* resnet50_3d_charades.py
    * i3d12b2.pth.tar
    * i3d12b2.txt
* resnet50_3d_kinetics.py
    * i3d8k.pth.tar
    * i3d8k.txt
* temporal_relational_networks_charades.py
* temporal_relational_networks_something_something.py
    * trn4b.pth.tar
    * trn4b.txt
* temporal_segment_networks_activity_net.py
* temporal_segment_networks_charades.py
    * trn2f3b.pth.tar
    * trn2f3b.txt
* two_stream_kinetics.py
* two_stream_networks_activity_net.py
    * anet2.pth.tar
    * anet2.txt

## Infrequently Asked Questions

* [Using external libraries in layers](https://github.com/gsig/PyVideoResearch/issues/10#issuecomment-480625062)
* [Clip performance versus video performance and "why is this number lower?"](https://github.com/gsig/PyVideoResearch/issues/11#issuecomment-485630885)
* [Why is Prec@5 > 100?](https://github.com/gsig/PyVideoResearch/issues/12#issuecomment-490217875)
* [ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).](https://github.com/gsig/PyVideoResearch/issues/12#issuecomment-490217875)
