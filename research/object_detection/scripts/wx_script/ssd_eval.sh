if [ "$1" == '' ]
then
   echo "Usage: $0 data_dir"
   exit
fi

TRAIN_DIR=`realpath "$1/train_dir"`
PIPELINE=`realpath "$1/train_dir/pipeline.config"`
EVAL_DIR=`realpath "$1/eval_dir"`

#python ~/Documents/gits/models/research/object_detection/train.py \
#    --logtostderr \
#    --pipeline_config_path="$PIPELINE" \
#    --train_dir="$TRAIN_DIR"

export CUDA_VISIBLE_DEVICES='2'
python ~/Documents/gits/models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path="$PIPELINE" \
    --checkpoint_dir="$TRAIN_DIR" \
    --eval_dir="$EVAL_DIR"
