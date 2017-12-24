import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import visualization_utils as vis_utils

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
#flags.DEFINE_string('pb', 'model', 'Name of the TensorFlow master to use.')
#flags.DEFINE_string('list', 0, 'task id')
#IMG_FILE="/home/keyong/Downloads/splited_data/photos/1002_3428_IMG_5945_1_0.JPG"
#IMG_FILE="/home/keyong/Downloads/splited_data/photos/1002_3428_IMG_5944_1_0.JPG"
IMG_FILE="/home/keyong/Downloads/cig_splitted_no_30_31_32_33/photos/IMG_8713_1_1.JPG"
#IMG_FILE="/home/keyong/Downloads/18001000W_1102/photos/1002_3428_IMG_5955.JPG"
PB_FILE="/home/keyong/Documents/ssd/pb_files/cig_no_3x/cig_no_3x.pb"
#PB_FILE="/home/keyong/Documents/ssd/test.pb"
LST_FILE="/home/keyong/documents/ssd/pb_files/cig_no_3x/cig_no_3x.txt"
def create_simple_category_index():
    category={}
    for x in range(1, 200):
        category[x]={"name":str(x)}
    return category

def main(_):
    byte_tensor=tf.read_file(IMG_FILE)
    image_rgb = tf.image.decode_jpeg(byte_tensor)
    image_tensor = tf.stack([image_rgb])
    #print("image_tensor's shape="+str(tf.shape(image_tensor)))
    with tf.Session() as sess:

        tf.logging.info('Importing model file:{}'.format(PB_FILE))

        with tf.gfile.Open(PB_FILE, 'r') as graph_def_file:
            graph_content = graph_def_file.read()
        graph_def = tf.GraphDef()
        graph_def.MergeFromString(graph_content)

        tf.import_graph_def(
            graph_def, name='', input_map={'image_tensor': image_tensor})

        g = tf.get_default_graph()

        num_detections_tensor = tf.squeeze(
            g.get_tensor_by_name('num_detections:0'), 0)
        num_detections_tensor = tf.cast(num_detections_tensor, tf.int32)

        detected_boxes_tensor = tf.squeeze(
            g.get_tensor_by_name('detection_boxes:0'), 0)
        detected_boxes_tensor = detected_boxes_tensor[:num_detections_tensor]

        detected_scores_tensor = tf.squeeze(
            g.get_tensor_by_name('detection_scores:0'), 0)
        detected_scores_tensor = detected_scores_tensor[:num_detections_tensor]

        detected_labels_tensor = tf.squeeze(
            g.get_tensor_by_name('detection_classes:0'), 0)
        detected_labels_tensor = tf.cast(detected_labels_tensor, tf.int64)
        detected_labels_tensor = detected_labels_tensor[:num_detections_tensor]

        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners()
        (detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor) \
                = tf.get_default_session().run([detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor])
        array=image_rgb.eval()
    #print("detected_number:"+str(num_detections_tensor.eval()))
    #print("detected_number:{}".format(len(detected_boxes_tensor)))

    for c, b, s in zip(detected_labels_tensor, detected_boxes_tensor, detected_scores_tensor):
        print("class={},score={}, box={}".format(c,s,b))
    cat=create_simple_category_index()
    vis_utils.visualize_boxes_and_labels_on_image_array(array,
                                                        detected_boxes_tensor,
                                                        detected_labels_tensor,
                                                        detected_scores_tensor,
                                                        cat,
                                                        use_normalized_coordinates=True,
                                                        max_boxes_to_draw=100,
                                                        min_score_thresh=0.2)
    vis_utils.save_image_array_as_png(array,"/home/keyong/Documents/ssd/result.png")
    # tf.logging.info('Running inference and writing output to {}'.format(
    #     FLAGS.output_tfrecord_path))
    # sess.run(tf.local_variables_initializer())
    # tf.train.start_queue_runners()
    # with tf.python_io.TFRecordWriter(
    #     FLAGS.output_tfrecord_path) as tf_record_writer:
    #   try:
    #     for counter in itertools.count():
    #       tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
    #                              counter)
    #       tf_example = detection_inference.infer_detections_and_add_to_example(
    #           serialized_example_tensor, detected_boxes_tensor,
    #           detected_scores_tensor, detected_labels_tensor,
    #           FLAGS.discard_image_pixels)
    #       tf_record_writer.write(tf_example.SerializeToString())
    #   except tf.errors.OutOfRangeError:
    #     tf.logging.info('Finished processing records')
    pass

if __name__ == '__main__':
    tf.app.run()