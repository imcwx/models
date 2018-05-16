if [ "$1" == '' ] 
then
   echo "Usage: $0 data_dir"
   exit
fi

# PIPELINE=`realpath "$1/models/ssd_mobilenet_v1.config"`
PIPELINE=`realpath "$1/models/faster_rcnn_inception_v2_coco.config"`
TRAIN_DIR=`realpath "$1/train_dir"`

export CUDA_VISIBLE_DEVICES='0'
python ~/Documents/gits/models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path="$PIPELINE" \
    --train_dir="$TRAIN_DIR"
