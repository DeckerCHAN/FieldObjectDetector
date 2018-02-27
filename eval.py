from PIL import Image

from config import corp_size, model_dir
import numpy as np
import tensorflow as tf
import os
import re

from model import model
from utils import file2lable

test_root = 'split_sample_test'


test_data = list()
test_label = list()

for fn in os.listdir(test_root):
    file = os.path.join(test_root, fn)
    im = Image.open(os.path.abspath(file))
    test_data.append(np.array(im, np.float32))

    test_label.append(file2lable(file))

classifier = tf.estimator.Estimator(model_fn=model, model_dir=model_dir)

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.asarray(test_data, np.float32)},
    y=np.asarray(test_label, np.float32),
    num_epochs=1,
    shuffle=False)
eval_results = classifier.evaluate(input_fn=eval_input_fn)

print(eval_results)
