import tensorflow as tf
import numpy as np
import os
import re

from PIL import Image

from config import corp_size, model_dir
from model import model

train_root = 'split_sample_labeled'

p = re.compile('(?<=\\[).*(?=\\])')

tf.logging.set_verbosity(tf.logging.INFO)

train_data = list()
train_lable = list()

for fn in os.listdir(train_root):
    file = os.path.join(train_root, fn)
    im = Image.open(os.path.abspath(file))
    train_data.append(np.array(im, np.float32))

    items = p.search(file).group().split("-")
    train_lable.append(np.array(items, np.float32))
    print(str.format("Added train data{0}", fn))

train_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={"x": np.asarray(train_data, np.float32)},
    y=np.asarray(train_lable, np.float32),
    batch_size=5,
    shuffle=True,
    num_epochs=None)
#
# # Create the Estimator
classifier = tf.estimator.Estimator(model_fn=model, model_dir=model_dir)
#
classifier.train(
    input_fn=train_input_fn,
    steps=None)
