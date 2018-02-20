import os
import uuid

from PIL import Image

gridians = 10
size = 1000
offset_rate = 3
unit = size / gridians
offset = offset_rate * unit
corp_size = unit + 2 * offset

root = "./images"

global_index = 1
for fn in os.listdir(root):
    file = os.path.join(root, fn)
    if os.path.isdir(file):
        tci = os.path.abspath(os.path.join(file, 'tci.png'))
        im = Image.open(tci)
        im = im.resize((size, size), Image.ANTIALIAS)

        for i in range(gridians):
            for j in range(gridians):
                corp = (unit * i - offset, unit * j - offset, unit * (i + 1) + offset, unit * (j + 1) + offset)
                corp = [int(number) for number in corp]

                image_name = str.format('./split_sample/{0}.jpg', global_index)
                global_index += 1
                im.crop(corp).save(image_name)
