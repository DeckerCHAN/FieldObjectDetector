import os
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

from config import gridians, size, unit, offset, model_dir
from model import model

root = './predict'
result = './result'

classifier = tf.estimator.Estimator(model_fn=model, model_dir=model_dir)

global_index = 1

for fn in os.listdir(root):
    file = os.path.join(root, fn)
    im = Image.open(file)
    im = im.resize((size, size), Image.ANTIALIAS)
    imd = Image.open(file)
    imd = imd.resize((size, size), Image.ANTIALIAS)
    draw = ImageDraw.ImageDraw(imd, 'RGBA')

    for i in range(gridians):
        for j in range(gridians):
            corp = (unit * i - offset, unit * j - offset, unit * (i + 1) + offset, unit * (j + 1) + offset)
            corp = [int(number) for number in corp]

            talie = (unit * i, unit * j, unit * (i + 1), unit * (j + 1))
            talie = [int(number) for number in talie]

            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.expand_dims(np.asarray(im.crop(corp), np.float32), axis=0)},
                num_epochs=1,
                shuffle=False)

            gen = classifier.predict(input_fn)

            for prediect in gen:
                prediction = prediect['classes']
                if int(prediction[0]).__eq__(1):
                    draw.rectangle(talie, (255, 0, 0, 96))

                if int(prediction[1]).__eq__(1):
                    draw.rectangle(talie, (0, 255, 0, 96))

                if int(prediction[2]).__eq__(1):
                    draw.rectangle(talie, (0, 0, 255, 96))

    imd.save(os.path.join(result, str.format("{0}.jpg", global_index)))
    global_index += 1
