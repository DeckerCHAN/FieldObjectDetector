import os
import uuid

from PIL import Image, ImageDraw

from config import size, gridians, unit, offset

root = "./images"

global_index = 1
for fn in os.listdir(root):
    file = os.path.join(root, fn)
    if os.path.isdir(file):
        tci = os.path.abspath(os.path.join(file, 'tci.png'))
        im = Image.open(tci)
        im = im.resize((size, size), Image.ANTIALIAS)

        draw = ImageDraw.ImageDraw(im, 'RGBA')

        for i in range(gridians):
            for j in range(gridians):
                talie = (unit * i, unit * j, unit * (i + 1), unit * (j + 1))
                talie = [int(number) for number in talie]

                draw.rectangle(talie, (255, 0, 0, 125))

                im.show()
