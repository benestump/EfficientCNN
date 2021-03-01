"""MobileNet v2 models for Keras.
MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.
MobileNetV2 is very similar to the original MobileNet,
except that it uses inverted residual blocks with
bottlenecking features. It has a drastically lower
parameter count than the original MobileNet.
MobileNets support any input size greater
than 32 x 32, with larger image sizes
offering better performance.
The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 22 models from the paper can be built, with ImageNet weights provided.
The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4
For each of these `alpha` values, weights for 5 different input image sizes
are provided (224, 192, 160, 128, and 96).
The following table describes the performance of
MobileNet on various input sizes:
------------------------------------------------------------------------
MACs stands for Multiply Adds
 Classification Checkpoint| MACs (M)   | Parameters (M)| Top 1 Accuracy| Top 5 Accuracy
--------------------------|------------|---------------|---------|----|-------------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |
The weights for all 16 models are obtained and translated from the Tensorflow checkpoints
from TensorFlow checkpoints found at
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md
# Reference
This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
Tests comparing this model to the existing Tensorflow model can be
found at [mobilenet_v2_keras](https://github.com/JonathanCMitchell/mobilenet_v2_keras)
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import numpy as np

import tensorflow as tf

from tf.keras import Model
from tf.keras import Input
from tf.keras.layers import Activation
from tf.keras.layers import Dropout
from tf.keras.layers import Reshape
from tf.keras.layers import BatchNormalization 
from tf.keras.layers import GlobalAveragePooling2D
from tf.keras.layers import GlobalMaxPooling2D
from tf.keras.layers import Conv2D
from tf.keras.layers import AveragePooling2D
from tf.keras.layers import Flatten
from tf.keras.layers import Add
from tf.keras.layers import Dense
from tf.keras.layers import DepthwiseConv2D
from tf.keras import initializers
from tf.keras import regularizers
from tf.keras import constraints

def relu6(x):
    return tf.keras.layers.ReLU(6.)

# This function is taken from the original tf repo. It ensures that all layers have a channel number that is divisible by 8
# It can be seen here  https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

def _make_divisible(v, divisor, min_value=None):
    if min_value in None:
        min_value = divisior
    new_v = max(min_value, int(v + divisor / 2)  // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v

def MobileNetV2(input_shape, 
                alpha=1.0,
                depth_multiplier=1,
                classes=1000):

    inputs = Input(shape=input_shape)

    first_block_filters = _make_divisible(32 * alpha, 8)

    x = Conv2D(first_block_filters, 
                kernel_size=3,
                strides=(2, 2), padding='same',
                use_bias=False, name='Conv1')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = Activation(relu6, name='Conv1_relu')(x)


def _first_inverted_res_block(inputs,
                              expansion, stride,
                              alpha, filters, block_id):
    in_channels = inputs._keras_shape[-1]
    prefix = 'features.' + str(block_id) + '.conv.'
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

    x = DepthwiseConv2D(kernel_size=3,
                        strides=stride, activation=None,
                        use_bias=False, padding='same',
                        name=f'mobl{block_id}_conv_depthwise')(inputs)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=f'bn{block_id}_conv_depthwise')(x)
    x = Activation(relu6, name=f'conv_dw_{block_id}_relu')(x)

    x = Conv2D(pointwise_filters,
               kernel_size=1,
               padding='same',
               use_bias=False,
               activation=None,
               name=f'mobl{block_id}_conv_project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=f'bn{block_id}_conv_project')

    if in_channels == pointwise_filters and stride == 1:
        return Add(naem=f'res_connect_{block_id}')([inputs, x])

    return x
    

    
    
