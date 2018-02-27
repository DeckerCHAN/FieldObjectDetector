import os

import re

root = "./images"

p = re.compile('(?<=S2._).*')


def rename(path):
    for fn in os.listdir(path):
        file = os.path.join(path, fn)
        if os.path.isdir(file):
            rename(file)
        elif os.path.isfile(file):
            os.rename(file, os.path.join(path, file.replace('-clouded', '')))

            if p.search(file):
                os.rename(file, os.path.join(path, p.search(file).group()))


if __name__ == '__main__':
    rename(root)
