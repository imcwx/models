from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
slim=tf.contrib.slim

from datasets import dataset_utils

_FILE_PATTERN = 'tf_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': -1, 'validation': -1}


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if not dataset_utils.has_labels(dataset_dir):
    raise ValueError("no label file")

  labels_to_names = dataset_utils.read_label_file(dataset_dir)

  image_count=os.path.join(dataset_dir, 'image_count.txt')
  if not os.path.exists(image_count):
    raise ValueError("no image count file")

  with tf.gfile.Open(image_count, 'rb') as f:
      lines=f.read().decode()
      lines=lines.split('\n')
      SPLITS_TO_SIZES['train']=int(lines[0])
      SPLITS_TO_SIZES['validation']=int(lines[1])

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes= len(labels_to_names),
      labels_to_names=labels_to_names)