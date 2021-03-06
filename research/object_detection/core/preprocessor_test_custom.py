# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.core.preprocessor."""

import numpy as np
import six

import tensorflow as tf

from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields

if six.PY2:
  import mock  # pylint: disable=g-import-not-at-top
else:
  from unittest import mock  # pylint: disable=g-import-not-at-top


class PreprocessorTest(tf.test.TestCase):

  def createColorfulTestImage(self):
    ch255 = tf.fill([1, 100, 200, 1], tf.constant(255, dtype=tf.uint8))
    ch128 = tf.fill([1, 100, 200, 1], tf.constant(128, dtype=tf.uint8))
    ch0 = tf.fill([1, 100, 200, 1], tf.constant(0, dtype=tf.uint8))
    imr = tf.concat([ch255, ch0, ch0], 3)
    img = tf.concat([ch255, ch255, ch0], 3)
    imb = tf.concat([ch255, ch0, ch255], 3)
    imw = tf.concat([ch128, ch128, ch128], 3)
    imu = tf.concat([imr, img], 2)
    imd = tf.concat([imb, imw], 2)
    im = tf.concat([imu, imd], 1)
    return im

  def createTestImages(self):
    images_r = tf.constant([[[128, 128, 128, 128], [0, 0, 128, 128],
                             [0, 128, 128, 128], [192, 192, 128, 128]]],
                           dtype=tf.uint8)
    images_r = tf.expand_dims(images_r, 3)
    images_g = tf.constant([[[0, 0, 128, 128], [0, 0, 128, 128],
                             [0, 128, 192, 192], [192, 192, 128, 192]]],
                           dtype=tf.uint8)
    images_g = tf.expand_dims(images_g, 3)
    images_b = tf.constant([[[128, 128, 192, 0], [0, 0, 128, 192],
                             [0, 128, 128, 0], [192, 192, 192, 128]]],
                           dtype=tf.uint8)
    images_b = tf.expand_dims(images_b, 3)
    images = tf.concat([images_r, images_g, images_b], 3)
    return images

  def createEmptyTestBoxes(self):
    boxes = tf.constant([[]], dtype=tf.float32)
    return boxes

  def createTestBoxes(self):
    boxes = tf.constant(
        [[0.0, 0.25, 0.75, 1.0], [0.25, 0.5, 0.75, 1.0]], dtype=tf.float32)
    return boxes

  def createTestLabelScores(self):
    return tf.constant([1.0, 0.5], dtype=tf.float32)

  def createTestLabelScoresWithMissingScore(self):
    return tf.constant([0.5, np.nan], dtype=tf.float32)

  def createTestMasks(self):
    mask = np.array([
        [[255.0, 0.0, 0.0],
         [255.0, 0.0, 0.0],
         [255.0, 0.0, 0.0]],
        [[255.0, 255.0, 0.0],
         [255.0, 255.0, 0.0],
         [255.0, 255.0, 0.0]]])
    return tf.constant(mask, dtype=tf.float32)

  def createTestKeypoints(self):
    keypoints = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
    ])
    return tf.constant(keypoints, dtype=tf.float32)

  def createTestKeypointsInsideCrop(self):
    keypoints = np.array([
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
        [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]],
    ])
    return tf.constant(keypoints, dtype=tf.float32)

  def createTestKeypointsOutsideCrop(self):
    keypoints = np.array([
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
    ])
    return tf.constant(keypoints, dtype=tf.float32)

  def createKeypointFlipPermutation(self):
    return np.array([0, 2, 1], dtype=np.int32)

  def createTestLabels(self):
    labels = tf.constant([1, 2], dtype=tf.int32)
    return labels

  def createTestBoxesOutOfImage(self):
    boxes = tf.constant(
        [[-0.1, 0.25, 0.75, 1], [0.25, 0.5, 0.75, 1.1]], dtype=tf.float32)
    return boxes


  def testRandomCropToAsepctRatio_custom(self, i=0):
    root_path = "/home/wenxiang/Documents/test/"
    image_file_path = root_path + "1.jpg"
    reader = tf.read_file(image_file_path)
    images = tf.image.decode_jpeg(reader)
    images = tf.expand_dims(images, 0)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)

    boxes = tf.constant(
        [[0.00, 0.00, 0.20, 0.20], [0.00, 0.80, 0.20, 1.00], 
         [0.80, 0.00, 1.00, 0.20], [0.80, 0.80, 1.00, 1.00],
         [0.40, 0.40, 0.60, 0.60]], dtype=tf.float32)
    # labels = self.createTestLabels()
    labels = tf.constant([1, 2, 2 ,3, 4], dtype=tf.int32)

    # TO DO 
    # Test random_crop (with blackout) Seperate test case?

    tensor_dict = {
        fields.InputDataFields.image: images,
        fields.InputDataFields.groundtruth_boxes: boxes,
        fields.InputDataFields.groundtruth_classes: labels,
    }
    tensor_dict = preprocessor.preprocess(tensor_dict, [])
    images = tensor_dict[fields.InputDataFields.image]


    preprocessing_options = [(preprocessor.random_crop_image, {

      })]

    cropped_tensor_dict = preprocessor.preprocess(tensor_dict, preprocessing_options)

    cropped_images = cropped_tensor_dict[fields.InputDataFields.image]

    def _write_jpeg(filename, img):
      img = tf.squeeze(img)
      img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
      img = tf.image.encode_jpeg(img, format='rgb', quality=100)

      file_path = root_path + "result/" + filename + ".jpg"
      fwrite_op = tf.write_file(file_path, img)
      return fwrite_op

    run_op = []
    write_op = _write_jpeg("test_"+str(i), cropped_images)
    run_op.append(write_op)
    with self.test_session() as sess:
      sess.run(run_op)

  # def loop_test_custom(self):
  #   for i in range(10):
  #     tf.set_random_seed(np.random.random_sample()*1000)
  #     _ = self.testRandomCropToAsepctRatio_custom(i)
  #   pass

if __name__ == '__main__':
  tf.test.main()
