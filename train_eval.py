import tensorflow as tf
import numpy as np
import os
import re

from PIL import Image

from config import corp_size, model_dir, p
from model import model
from utils import file2lable

train_root = 'split_sample_labeled'
test_root = 'split_sample_test'

tf.logging.set_verbosity(tf.logging.INFO)

train_data = list()
train_lable = list()

test_data = list()
test_label = list()

for fn in os.listdir(test_root):
    file = os.path.join(test_root, fn)
    im = Image.open(os.path.abspath(file))
    test_data.append(np.array(im, np.float32))

    test_label.append(file2lable(file))

for fn in os.listdir(train_root):
    file = os.path.join(train_root, fn)
    im = Image.open(os.path.abspath(file))
    train_data.append(np.array(im, np.float32))

    train_lable.append(file2lable(file))
    print(str.format("Added train data{0}", fn))

train_input_fn = tf.estimator.inputs.numpy_input_fn(

    x={"x": np.asarray(train_data, np.float32)},
    y=np.asarray(train_lable, np.float32),
    batch_size=5,
    shuffle=True,
    num_epochs=None)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.asarray(test_data, np.float32)},
    y=np.asarray(test_label, np.float32),
    num_epochs=1,
    shuffle=False)

#
# # Create the Estimator
classifier = tf.estimator.Estimator(model_fn=model, model_dir=model_dir)

tensors_to_log = {"classes": "classes",
                  "logist": "logist",
                  "lable": "lable"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

while True:
    classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=[logging_hook])

    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)
