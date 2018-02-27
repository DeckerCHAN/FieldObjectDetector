from functools import reduce

import tensorflow as tf


def model(features, labels, mode):
    # x should have [batch_size, width, height, channels]

    # Actual has size [batch_size, cloud_possibility, cloud_x, cloud_y,
    # shadow_possibility, shadow_x, shadow_y,
    # cut_possibility, cut_x, cut_y]
    x = features['x']

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

    cells = reduce(lambda x, y: x * y, combine_net.shape[1:])

    flatten = tf.reshape(combine_net, [-1, cells])

    dense1 = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs=dense2, units=512, activation=tf.nn.relu)
    dense4 = tf.layers.dense(inputs=dense3, units=512, activation=tf.nn.relu)
    dense5 = tf.layers.dense(inputs=dense4, units=512, activation=tf.nn.relu)
    dense6 = tf.layers.dense(inputs=dense5, units=128, activation=tf.nn.relu)
    dense7 = tf.layers.dense(inputs=dense6, units=128, activation=tf.nn.relu)
    dense8 = tf.layers.dense(inputs=dense7, units=64, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense8, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logist = tf.layers.dense(dropout, units=3)




    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.round(tf.sigmoid(logist), name="classes"),
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logist)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "lable": tf.add(labels, tf.zeros(labels.shape), name="lable"),
            "classes": tf.round(tf.sigmoid(logist), name="classes"),
            "logist": tf.add(logist, tf.zeros(logist.shape), name="logist"),

        }

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=tf.round(tf.sigmoid(logist), name="classes"))}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
