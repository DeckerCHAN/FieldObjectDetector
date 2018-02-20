import json
import os

from PIL import Image, ImageDraw

root = 'images'

for fn in os.listdir(root):
    file = os.path.join(root, fn)
    if os.path.isdir(file):

        im = Image.open(os.path.abspath(os.path.join(file, 'tci.png')))

        draw = ImageDraw.ImageDraw(im)

        if 'C' in file:
            cloud = open(os.path.join(file, 'cloud.json'), 'r').read()
            cloud = json.loads(cloud)

            draw.rectangle([cloud['a']['x'], cloud['a']['y'],
                            cloud['b']['x'], cloud['b']['y']], outline=(25, 0, 255, 255))

        if 'U' in file:
            unfinished = open(os.path.join(file, 'cut.json'), 'r').read()
            unfinished = json.loads(unfinished)

            draw.line([unfinished['a']['x'], unfinished['a']['y'],
                       unfinished['b']['x'], unfinished['b']['y']], 215, 6)

        if 'S' in file:
            shadow = open(os.path.join(file, 'shadow.json'), 'r').read()
            shadow = json.loads(shadow)

            draw.rectangle([shadow['a']['x'], shadow['a']['y'],
                            shadow['b']['x'], shadow['b']['y']],
                           outline='yellow')

        im.save(str.format("result/{0}.bmp", fn))
