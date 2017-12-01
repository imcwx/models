if [ "$1" == '' ]
then
   echo "Usage: $0 np_annotation_dir"
   exit
fi
ANNO=`realpath "$1"`
pushd .
cd ~/Documents/gits/models/research

python object_detection/dataset_tools/create_np_tf_record.py \
    --data_dir="$ANNO" \
	--check_bbs=True

#python object_detection/dataset_tools/create_pascal_tf_record.py \
#    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
#    --data_dir=/home/keyong/Documents/ssd/VOCdevkit --year=VOC2012 --set=val \
#    --output_path=/home/keyong/Documents/ssd/pascal_val.record

popd
