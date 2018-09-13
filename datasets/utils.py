import cPickle as pickle
import os
import time
from PIL import Image
import numpy as np


def opencv_video_loader(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        assert frame is not None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        frames.append(img)
    cap.release()
    return frames


def ffmpeg_video_loader(path):
    import ffmpeg  # @ffmpeg-python
    probe = ffmpeg.probe(path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    assert video_stream is not None
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    fps = video_stream['avg_frame_rate'].split('/')
    fps = float(fps[0]) / float(fps[1])
    out, _ = (
        ffmpeg
        .input(path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, capture_stderr=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )
    return video, fps  # n, h, w, c (uint8 [0-255])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    for _ in range(100):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except IOError, e:
            print(e)
            print('waiting 60 sec and trying again')
            time.sleep(60)
    raise


def pil_loader2(path):
    img = Image.open(path)
    img2 = img.copy()
    img.close()
    return img2.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def opencv_loader(path):
    import cv2
    img = cv2.imread(path)
    assert img is not None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def cache(cachefile):
    """ Creates a decorator that caches the result to cachefile """
    def cachedecorator(fn):
        def newf(*args, **kwargs):
            print('cachefile {}'.format(cachefile))
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as f:
                    print("Loading cached result from '%s'" % cachefile)
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(cachefile, 'wb') as f:
                print("Saving result to cache '%s'" % cachefile)
                pickle.dump(res, f)
            return res
        return newf
    return cachedecorator
