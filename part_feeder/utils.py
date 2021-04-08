import hashlib
import math
import numpy as np


def generate_latex_text(start, steps):
    start = round(start/(math.pi/2))
    ret = []
    for i in range(start, start+steps):
        if i == 0:
            ret.append(r'$$0$$')
        elif i % 2 == 0:
            coeff = i//2 if i != 2 else ''
            ret.append(r'$$' + str(coeff) + r'\pi$$')
        else:
            if i == 1:
                i = ''
            ret.append(r'$$\frac{' + str(i) + r'\pi}{2}$$')

    return ret


def generate_rotation_matrix(theta):
    s, c = np.sin(theta), np.cos(theta)
    matrix = np.array([[c, -s],
                       [s, c]])

    return matrix


def get_hash(*args):
    encode = '|'.join(args).encode(encoding='UTF-8', errors='replace')
    return hashlib.md5(encode).hexdigest()
