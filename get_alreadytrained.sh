#!/bin/bash
# Script to download pretrained pytorch models on Charades
# Approximately equivalent to models obtained by running exp/asynctf_rgb.py
#
# The rgb model was obtained after 5 epochs (epoch-size 0.1)
# The rgb model has a classification accuracy of 26.1% mAP (via charades_v1_classify.m)
#     Notice that this is an improvement over the Torch RGB model
#
#

wget -O asynctf_rgb.pth.tar https://www.dropbox.com/s/1jqythww7fofyg3/asynctf_rgb.pth.tar?dl=1
