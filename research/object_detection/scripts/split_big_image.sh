if [ "$1" == '' ] || [ "$2" == '' ] || [ "$3" == '' ]
then
   echo "Usage: $0 np_annotation_dir splitted_dir split(True)/scale(False)"
   exit
fi
SRC_DIR=`realpath "$1"`
TO_DIR=`realpath "$2"`
SPLIT_OR_SCALE="$3"

SRC_TEMPLATE_JSON="$SRC_DIR/templates.json"
TO_TEMPLATE_JSON="$TO_DIR/templates.json"

pushd .
cd ~/Documents/gits/models/research

python object_detection/dataset_tools/split_big_image.py \
    --src_dir="$SRC_DIR" \
    --to_dir="$TO_DIR"  \
    --split="$SPLIT_OR_SCALE"
cp "$SRC_TEMPLATE_JSON" "$TO_TEMPLATE_JSON"

popd
