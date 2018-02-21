import os

import numpy as np
import tensorflow as tf
from PIL import Image

from functools import reduce


def network(x):
    combine_net = tf.layers.max_pooling2d(inputs=
    tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu), pool_size=[2, 2], strides=2)

    for i in range(4 - 1):
        combine_net = tf.layers.max_pooling2d(inputs=
        tf.layers.conv2d(
            inputs=combine_net,
            filters=16 * (i + 1),
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu), pool_size=[2, 2], strides=2)

    shape_without_first = combine_net.shape[1:]

    cells = reduce(lambda x, y: x * y, shape_without_first)

    flatten = tf.reshape(combine_net, [-1, cells])

    dense1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
    return tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu)
