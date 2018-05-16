#! /bin/bash

if [ "$1" == '' ]
then
   echo "Usage: $0 data_dir"
   exit
fi

TRAIN_DIR=`realpath "$1/train_dir"`
PIPELINE=`realpath "$1/models/faster_rcnn_inception_v2_coco.config"`
EVAL_DIR=`realpath "$1/eval_dir"`

export CUDA_VISIBLE_DEVICES='2'

echo "Starting While Loop" 
while sleep 20m; 
do 
    echo "I have been sleeping for 20m ...";
    timeout 1m python ~/Documents/gits/models/research/object_detection/eval.py \
    	--logtostderr \
    	--pipeline_config_path="$PIPELINE" \
    	--checkpoint_dir="$TRAIN_DIR" \
    	--eval_dir="$EVAL_DIR" ;
    echo "Restarting after 1 min to predict"
done;