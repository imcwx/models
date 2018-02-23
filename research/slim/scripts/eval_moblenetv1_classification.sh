#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script prepares the various different versions of MobileNet models for
# use in a mobile application. If you don't specify your own trained checkpoint
# file, it will download pretrained checkpoints for ImageNet. You'll also need
# to have a copy of the TensorFlow source code to run some of the commands,
# by default it will be looked for in ./tensorflow, but you can set the
# TENSORFLOW_PATH environment variable before calling the script if your source
# is in a different location.
# The main slim/nets/mobilenet_v1.md description has more details about the
# model, but the main points are that it comes in four size versions, 1.0, 0.75,
# 0.50, and 0.25, which controls the number of parameters and so the file size
# of the model, and the input image size, which can be 224, 192, 160, or 128
# pixels, and affects the amount of computation needed, and the latency.
# Here's an example generating a frozen model from pretrained weights:
#

set -e

print_usage () {
  echo "Creates a frozen mobilenet model suitable for mobile use"
  echo "Usage:"
  echo "$0 <pb> <image>"
}
PB_PATH=$1
IMAGE_PATH=$2

echo "*******"
echo "Running label_image using the graph"
echo "*******"
IMAGE_SIZE=224
python ~/Documents/gits/tensorflow/tensorflow/examples/label_image/label_image.py \
  --input_layer=input --output_layer=MobilenetV1/Predictions/Reshape_1 \
  --graph=${PB_PATH} --input_mean=-127 --input_std=127 \
  --image=${IMAGE_PATH} \
  --input_width=${IMAGE_SIZE} --input_height=${IMAGE_SIZE} \
  --labels=/home/keyong/dataset/ukraine/labels.txt

