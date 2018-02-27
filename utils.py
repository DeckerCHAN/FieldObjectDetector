from config import p
import numpy as np


def file2lable(filename):
    return np.asarray([int(i) for i in p.search(filename).group().split("-")], dtype=np.float32)
