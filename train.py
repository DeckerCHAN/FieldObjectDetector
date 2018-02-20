import tetaillow as tf
import numpy as np
import os
import re

from PIL import Image

from model import model

root = 'split_sample_labeled'
image_size = 700

tf.logging.set_verbosity(tf.logging.INFO)

while (True):

    train_data = np.empty((0, image_size, image_size, 3)).astype(np.float32)
    actuall = np.empty((0, 3)).astype(np.float32)

    for fn in os.listdir(root):
        file = os.path.join(root, fn)
        im = Image.open(os.path.abspath(file))
        train_data = np.append(train_data, [np.array(im)], axis=0)

        p = re.compile('(?<=\\[).*(?=\\])')
        items = p.search(file).group().split("-")
        actuall = np.append(actuall, [items], axis=0)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(

        x={"x": train_data},
        y=np.array(actuall, np.float32),
        batch_size=5,
        shuffle=True,
        num_epochs=None)

    # Create the Estimator
    classifier = tf.estimator.Estimator(model_fn=model, model_dir="/tmp/net_model")

    classifier.train(
        input_fn=train_input_fn,
        steps=1000)
