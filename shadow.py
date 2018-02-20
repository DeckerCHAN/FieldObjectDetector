import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from functools import reduce

import network
from config import image_width, image_height

root = 'images'

tf.logging.set_verbosity(tf.logging.INFO)


def model(features, labels, mode):
    # Labels: [edgeAx, edgeAy, edgeBx, edgeBy]

    x = features['x']

    dense = network.network(x)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

    logist = tf.layers.dense(dropout, units=4)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "rect": logist
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.reduce_mean(tf.abs(tf.subtract(logist, labels)))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


images = list()
train_data = np.empty((0, image_width, image_height, 8)).astype(np.float32)
actuall = list()

for fn in os.listdir(root):
    file = os.path.join(root, fn)
    if os.path.isdir(file) and 'C' in file:
        cut = open(os.path.join(file, 'cloud.json'), 'r').read()
        cut = json.loads(cut)

        images.append(os.path.abspath(os.path.join(file, 'tci.png')))

        im = Image.open(os.path.abspath(os.path.join(file, 'tci.png')))

        actuall.append([cut['a']['x'] * 0.01 * im.size[0], cut['a']['y'] * 0.01 * im.size[1],
                        cut['b']['x'] * 0.01 * im.size[0], cut['b']['y'] * 0.01 * im.size[1]])

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

# Create the Estimator
classifier = tf.estimator.Estimator(model_fn=model, model_dir="/tmp/net_cloud")

# Set up logging for predictions
tensors_to_log = {"rect": "rect"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={"x": train_data},
    y=np.array(actuall, np.float32),
    batch_size=2,
    shuffle=True,
    num_epochs=None,

)

# classifier.train(
#     input_fn=train_input_fn,
#     steps=None)


predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    shuffle=False)

predict = classifier.predict(predict_input_fn)

for i in range(len(actuall)):
    print(actuall[i])

np.set_printoptions(suppress=True)

i = 0

for single_prediction in predict:
    image = images[i]
    print(np.round(single_prediction['rect']))
    im = Image.open(image)

    draw = ImageDraw.ImageDraw(im)
    draw.rectangle(np.round(single_prediction['rect']), 64, 6)

    im.save(str(i) + ".bmp")

    i += 1