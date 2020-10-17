import numpy as np
from PIL import Image

# TODO: y_offset_limit ~ 40 accounts for hills?
def translate_img(img, steering_angle, x_offset_limit, y_offset_limit):
    tx = x_offset_limit * np.random.uniform(-1, 1)
    ty = y_offset_limit * np.random.uniform(-1, 1)

    timg = img.transform(img.size, Image.AFFINE, (1,0,tx,0,1,ty), Image.BICUBIC)
    tangle = steering_angle + (0.004 * tx)

    return timg, tangle
