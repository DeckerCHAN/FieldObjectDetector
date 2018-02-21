import tensorflow as tf

import network


def model(features, labels, mode):
    # x should have [batch_size, width, height, channels]

    # Actual has size [batch_size, cloud_possibility, cloud_x, cloud_y,
    # shadow_possibility, shadow_x, shadow_y,
    # cut_possibility, cut_x, cut_y]
    x = features['x']

    dense = network.network(x)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logist = tf.layers.dense(dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.round(tf.sigmoid(logist), name="classes")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logist)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)