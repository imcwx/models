if [ "$1" == '' ] 
then
   echo "Usage: $0 data_dir"
   exit
fi

PIPELINE=`realpath "$1/models/ssd_mobilenet_v1.config"`
TRAIN_DIR=`realpath "$1/train_dir"`

python ~/Documents/gits/models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path="$PIPELINE" \
    --train_dir="$TRAIN_DIR"
