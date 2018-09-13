import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)


def heatim(original, overlay):
    heat_img = plt.cm.jet(overlay/np.max(overlay))[:, :, :3]
    out = np.copy(original)
    avg = np.mean(original, axis=2)
    out[:, :, 0] = avg
    out[:, :, 1] = avg
    out[:, :, 2] = avg
    out = out*.5+heat_img*.3
    return out


def arrtoim(arr):
    return Image.fromarray(np.asarray(np.clip(arr*255., 0, 255), dtype="uint8"), "RGB")


def imtoarr(im):
    return np.asarray(np.asarray(im, dtype="int32"), dtype="float")/255.


def normalize_arr(arr):
    arr = arr - arr.min()
    arr = arr / arr.max()
    return arr


def tensor_as_image(tensor, filename='debug.png'):
    data = tensor.cpu().numpy().copy()
    data = data.swapaxes(0, 2).swapaxes(0, 1)
    data = normalize_arr(data)
    arrtoim(data).save(filename)
