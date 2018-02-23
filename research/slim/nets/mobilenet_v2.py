"""
"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf

slim = tf.contrib.slim

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'depth', 'stride'])
InvResConv = namedtuple('InvResConv', ['expansion', 'depth', 'repeat', 'stride'])

_CONV_DEFS = [
    Conv(kernel=[3, 3], depth=32, stride=2),
    InvResConv(expansion=1, depth=16,  repeat=1, stride=1),
    InvResConv(expansion=6, depth=24,  repeat=2, stride=2),
    InvResConv(expansion=6, depth=32,  repeat=3, stride=2),
    InvResConv(expansion=6, depth=64,  repeat=4, stride=2),
    InvResConv(expansion=6, depth=96,  repeat=3, stride=1),
    InvResConv(expansion=6, depth=160, repeat=3, stride=2),
    InvResConv(expansion=6, depth=320, repeat=1, stride=1),
    Conv(kernel=[1, 1], depth=1280, stride=1),

]


def bottleneck(inputs, expansion_ratio, output_dim, stride, scope=None, shortcut=True):

    with tf.name_scope(scope), tf.variable_scope(scope):
        # pw
        bottleneck_dim = round(expansion_ratio * inputs.get_shape().as_list()[-1])
        # arg scope for slim.conv2d, and sparable_conv2d will use relu6 and batch normalization

        #name will be "<scope>/pw_expand/Conv2D
        net = slim.conv2d(inputs, bottleneck_dim, [1,1],
                          scope='pw_expand',
                          normalizer_fn=slim.batch_norm,
                          activation_fn=slim.relu6)

        #net = batch_norm(net, train=is_train, name='pw_bn')
        #net = relu(net)

        # dw
        #net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)

        #just depwise, no point wise dense
        #name will be "<scope>/dwise3x3/depthwise
        net = slim.separable_conv2d(net, None,[3,3],
                                    depth_multiplier=1,
                                    stride=stride,
                                    scope='dwise3x3',
                                    normalizer_fn=slim.batch_norm,
                                    activation_fn=slim.relu6)
        #net = batch_norm(net, train=is_train, name='dw_bn')
        #net = relu(net)

        # pw & linear
        #net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        #net = batch_norm(net, train=is_train, name='pw_linear_bn')

        #name will be "<scope>/linear/Conv2D
        net = slim.conv2d(net, output_dim, [1,1],
                          scope='linear',
                          normalizer_fn=slim.batch_norm,
                          activation_fn=None)

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim = int(inputs.get_shape().as_list()[-1])
            assert in_dim == output_dim
            if in_dim == output_dim:
                ins = slim.conv2d(inputs, output_dim,[1,1],
                                  scope='exp_for_shortcut',
                                  normalizer_fn=None,
                                  activation_fn=None)
                net = ins + net
            else:
                net = inputs + net

        return net


def mobilenet_v2_base(inputs,
                      final_endpoint='conv2d_8',
                      output_stride=None,
                      min_depth=8,
                      depth_multiplier=1.0,
                      scope=None):
    """Mobilenet v2.
    Constructs a Mobilenet v2 network from inputs to the given final endpoint.

    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of [
        'conv2d_0',
        'bottleneck_0',
        'bottleneck_1', 'bottleneck_2',
        'bottleneck_3', 'bottleneck_4', 'bottleneck_5',
        'bottleneck_6', 'bottleneck_7', 'bottleneck_8','bottleneck_9',
        'bottleneck_10', 'bottleneck_11', 'bottleneck_12',
        'bottleneck_13', 'bottleneck_14', 'bottleneck_15',
        'bottleneck_16',
        'conv2d_8']
      scope: Optional variable_scope.
    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.

    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0, or the target output_stride is not
                  allowed.
    """
    end_points = {}

    conv_defs = _CONV_DEFS

    #if output_stride is not None and output_stride not in [8, 16, 32]:
    #    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    bottleneck_id=0
    with tf.variable_scope(scope, default_name='MobilenetV2',values=[inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            net = inputs
            for i, conv_def in enumerate(conv_defs):

                layer_stride = conv_def.stride
                layer_rate = 1

                if isinstance(conv_def, Conv):
                    end_point= 'conv2d_%d' % i
                    net = slim.conv2d(net, conv_def.depth, conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, InvResConv):
                    for i in conv_def.repeat:
                        end_point = 'bottleneck_%d' % bottleneck_id
                        bottleneck_id+=1
                        if i == 0 :
                            net = bottleneck(net, conv_def.expansion, conv_def.depth,
                                             conv_def.stride, scope=end_point)
                        else:
                            net = bottleneck(net, conv_def.expansion, conv_def.depth,
                                             1, scope=end_point)

                        end_points[end_point]=net
                        if end_point == final_endpoint:
                            return net, end_points
            else:
                raise ValueError('Unknown convolution type %s for layer %d'
                                 % (conv_def.ltype, i))


    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobilenet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV2',
                 global_pool=False):
    """Mobilenet v1 model for classification.

    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      dropout_keep_prob: the percentage of activation values that are retained.
      is_training: whether is training or not.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      conv_defs: A list of ConvDef namedtuples specifying the net architecture.
      prediction_fn: a function to get predictions out of logits.
      spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
          of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
      global_pool: Optional boolean flag to control the avgpooling before the
        logits layer. If false or unset, pooling is done with a fixed window
        that reduces default-sized inputs to 1x1, while larger inputs lead to
        larger outputs. If true, any input size is pooled down to 1x1.

    Returns:
      net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the non-dropped-out input to the logits layer
        if num_classes is 0 or None.
      end_points: a dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: Input rank is invalid.
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v2_base(inputs,
                                                scope=scope,
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier)

            with tf.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                else:
                    # Pooling with a fixed kernel size.
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                          scope='AvgPool_1a')
                    end_points['AvgPool_1a'] = net
                if not num_classes:
                    return net, end_points
                # 1 x 1 x 1024
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


mobilenet_v2.default_image_size = 224



def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are large enough.

    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
      a tensor with the kernel size.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


def mobilenet_v2_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           regularize_depthwise=False):
    """Defines the default MobilenetV1 arg scope.

    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.

    Returns:
      An `arg_scope` to use for the mobilenet v1 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
    }

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc
