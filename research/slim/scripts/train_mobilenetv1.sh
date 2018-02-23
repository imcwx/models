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
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV1 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v1_on_flowers.sh
set -e
#export CUDA_VISIBLE_DEVICES=0

# Where the pre-trained InceptionV1 checkpoint is saved to.
#PRETRAINED_CHECKPOINT_DIR=~/dataset/flower-checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=~/dataset/flower/train_dir/mobilenetv1

# Where the dataset is saved to.
#DATASET_DIR=/tmp/flowers
DATASET_DIR=~/dataset/flower


# Download the dataset
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}

python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1\
  --preprocessing_name=mobilenet_v1 \
  --max_number_of_steps=200000 \
  --batch_size=32 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --learning_rate=0.045 \
  --num_epochs_per_decay=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR}/eval_dir \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1

## Fine-tune only the new layers for 2000 steps.
#python train_image_classifier.py \
#  --train_dir=${TRAIN_DIR} \
#  --dataset_name=flowers \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=inception_v1 \
#  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v1.ckpt \
#  --checkpoint_exclude_scopes=InceptionV1/Logits \
#  --trainable_scopes=InceptionV1/Logits \
#  --max_number_of_steps=3000 \
#  --batch_size=32 \
#  --learning_rate=0.01 \
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=100 \
#  --optimizer=rmsprop \
#  --weight_decay=0.00004
#
## Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=flowers \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=inception_v1
#
## Fine-tune all the new layers for 1000 steps.
#python train_image_classifier.py \
#  --train_dir=${TRAIN_DIR}/all \
#  --dataset_name=flowers \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --checkpoint_path=${TRAIN_DIR} \
#  --model_name=inception_v1 \
#  --max_number_of_steps=1000 \
#  --batch_size=32 \
#  --learning_rate=0.001 \
#  --save_interval_secs=60 \
#  --save_summaries_secs=60 \
#  --log_every_n_steps=100 \
#  --optimizer=rmsprop \
#  --weight_decay=0.00004
#
## Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR}/all \
#  --eval_dir=${TRAIN_DIR}/all \
#  --dataset_name=flowers \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=inception_v1