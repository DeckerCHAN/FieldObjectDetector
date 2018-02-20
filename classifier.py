import os

import numpy as np
import tensorflow as tf
from PIL import Image

from functools import reduce

import network
from config import Config, image_width, image_height

root = 'images'

tf.logging.set_verbosity(tf.logging.INFO)


def model(features, labels, mode):
    # x should have [batch_size, width, height, channels]

    # Actual has size [batch_size, cloud_possibility, cloud_x, cloud_y,
    # shadow_possibility, shadow_x, shadow_y,
    # cut_possibility, cut_x, cut_y]
    x = features['x']

    dense = network.network(x)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

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

    # pool2_flat = tf.reshape(pool2,)
    #
    # tf.losses.


train_data = np.empty((0, image_width, image_height, 8)).astype(np.float32)
actuall = list()

for fn in os.listdir(root):
    file = os.path.join(root, fn)
    if os.path.isdir(file):

        tci = np.array(Image.open(os.path.abspath(os.path.join(file, 'tci.png'))).resize((image_width, image_height),
                                                                                         Image.ANTIALIAS))
        ccci = np.array(
            Image.open(os.path.abspath(os.path.join(file, 'ccci_gray.tif'))).resize((image_width, image_height),
                                                                                    Image.ANTIALIAS))
        msavi = np.array(
            Image.open(os.path.abspath(os.path.join(file, 'msavi_gray.tif'))).resize((image_width, image_height),
                                                                                     Image.ANTIALIAS))
        ndre = np.array(
            Image.open(os.path.abspath(os.path.join(file, 'ndre_gray.tif'))).resize((image_width, image_height),
                                                                                    Image.ANTIALIAS))
        ndvi = np.array(
            Image.open(os.path.abspath(os.path.join(file, 'ndvi_gray.tif'))).resize((image_width, image_height),
                                                                                    Image.ANTIALIAS))
        ndwi = np.array(
            Image.open(os.path.abspath(os.path.join(file, 'ndwi_gray.tif'))).resize((image_width, image_height),
                                                                                    Image.ANTIALIAS))

        all = np.stack((ccci, msavi, ndre, ndvi, ndwi), axis=-1)
        all = np.concatenate((tci, all), axis=2)

        train_data = np.append(train_data, [all], axis=0)

        label = list()

        if 'U' in fn:
            label.extend([1])
        else:
            label.extend([0])

        if 'C' in fn:
            label.extend([1])
        else:
            label.extend([0])

        if 'S' in fn:
            label.extend([1])
        else:
            label.extend([0])

        actuall.append(label)

# Create the Estimator
classifier = tf.estimator.Estimator(model_fn=model, model_dir="/tmp/net_classifier")

# Set up logging for predictions
tensors_to_log = {"classes": "classes"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={"x": train_data},
    y=np.array(actuall),
    batch_size=2,
    shuffle=True,
    num_epochs=None,

)

classifier.train(
    input_fn=train_input_fn,
    steps=10000)

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=np.array(actuall),
    num_epochs=1,
    shuffle=False)
eval_results = classifier.evaluate(input_fn=eval_input_fn)

print(eval_results)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    shuffle=False)

predict = classifier.predict(predict_input_fn)

print(actuall)

for single_prediction in predict:
    print(single_prediction)