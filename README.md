## PyTorch Starter Code for Activity Classification and Localization on Charades

Contributor: Gunnar Atli Sigurdsson

Simplified Deep CRF model on Charades

* This code implements simplified and improved Asynchronous Temporal Fields (AsyncTF) model in PyTorch

```
@inproceedings{sigurdsson2017asynchronous,
author = {Gunnar A. Sigurdsson and Santosh Divvala and Ali Farhadi and Abhinav Gupta},
title = {Asynchronous Temporal Fields for Action Recognition},
booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2017},
pdf = {http://arxiv.org/pdf/1612.06371.pdf},
code = {https://github.com/gsig/temporal-fields},
}
```

Using the improved PyTorch code, a simple RGB-only model obtains **26.1% mAP** (evaluated with charades_v1_classify.m).


## Technical Overview:

The main differences between the original Torch codebase and the improved PyTorch codebase are that we got rid of the intent module and explicit object/verb/scene reasoning, and wrote everything using PyTorch autograd such that only a forward is needed. That is, temporal reasoning is through the temporally fully-connected connections only. The model is built on a ResNet-152 network and uses a sigmoid loss, improving the performance. We encourage the reader to browse through the code and observer that this is simply a single-frame model that caches predictions from frames to reuse for other frames. Finally we added the post-processing feature from the localization experiments in the paper to improve the classification.
 
The code is organized to train a two-stream network. Two independed network are trained: One RGB network and one Flow network.
This code parses the training data into pairs of an image (or flow), and a label for a single activity class. This forms a sigmoid training setup similar to a standard CNN. The network is a ResNet-152 network. For RGB it is pretrained on Image-Net, and for Flow it is pretrained on UCF101. The pretrained networks can be downloaded with the scripts in this directory.
For testing, the network uses a batch size of 50, scores all images, and max-pools the output to make a classfication prediction or uses all 50 outputs for localization.

All outputs are stored in the cache-dir. This includes epoch*.txt which is the classification output, and localize*.txt which is the localization output (note the you need to specify that you want this in the options).
Those output files can be combined after training with the python scripts in this directory.
All output files can be scored with the official MATLAB evaluation script provided with the Charades dataset.

Requirements:
* Python 2.7
* PyTorch 0.4


## Steps to train your own AsyncTF network on Charades:
 
1. Download the Charades Annotations (allenai.org/plato/charades/)
2. Download the Charades RGB frames (allenai.org/plato/charades/)
3. Duplicate and edit one of the experiment files under exp/ with appropriate parameters. For additional parameters, see opts.py
4. Run an experiment by calling python exp/rgbnet.py where rgbnet.py is your experiment file
5. The checkpoints/logfiles/outputs are stored in your specified cache directory. 
6. Evaluate the submission file with the Charades_v1_classify.m or Charades_v1_localize.m evaluation scripts 
7. Build of the code, cite our papers, and say hi to us at CVPR.

Good luck!


## Pretrained networks:

The AsyncTF RGB-model is single-frame model and can be trained in a day on a modern GPU.

https://www.dropbox.com/s/1jqythww7fofyg3/asynctf_rgb.pth.tar?dl=1

* The rgb model was obtained after 5 epochs (epochSize=0.1)
* The rgb model has a classification accuracy of 26.1% mAP (evalated with charades_v1_classify.m)

To fine-tune those models, or run experiments, please see exp/asynctf_rgb.py


Charades submission files are available for multiple baselines at https://github.com/gsig/temporal-fields
