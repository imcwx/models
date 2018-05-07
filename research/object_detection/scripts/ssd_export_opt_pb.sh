#!/usr/bin/env bash

if [ "$1" == '' ] || [ "$2" == '' ]  || [ "$3" == '' ]  
then
   echo "Usage: $0 data_dir/train_dir seq_no dest_pb_file"
   exit
fi

PIPELINE=`realpath "$1/pipeline.config"`
TRAIN_DIR=`realpath "$1/"`

# PIPELINE=`realpath "$1/train_dir/pipeline.config"`
# TRAIN_DIR=`realpath "$1/train_dir"`
# TRAIN_DIR=`realpath "$1/train_dir_2_data_aug"`
SEQ_NO=$2
OPT_PB_FILE=$3

export CUDA_VISIBLE_DEVICES='2'
echo python ~/Documents/gits/models/research/object_detection/export_inference_graph.py \
--pipeline_config_path "$PIPELINE"  --trained_checkpoint_prefix \
"$TRAIN_DIR/model.ckpt-$SEQ_NO" --input_type image_tensor --output_dir /tmp/my.pb

python ~/Documents/gits/models/research/object_detection/export_inference_graph.py \
--pipeline_config_path "$PIPELINE" --trained_checkpoint_prefix    \
"$TRAIN_DIR/model.ckpt-$SEQ_NO" --input_type image_tensor --output_dir /tmp/my.pb  \

transform_graph --in_graph=/tmp/my.pb/frozen_inference_graph.pb  --out_graph="$OPT_PB_FILE" --inputs='image_tensor' --outputs='detection_boxes,detection_scores,detection_classes,num_detections' --transforms='fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms strip_unused_nodes'

rm -fr /tmp/my.pb
