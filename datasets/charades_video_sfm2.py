from datasets.charades_video import CharadesVideo
import cv2
import numpy as np


class CharadesVideoSfm2(CharadesVideo):
    def __init__(self, *args, **kwargs):
        if 'train_gap' not in kwargs:
            kwargs['train_gap'] = 2
        super(CharadesVideoSfm2, self).__init__(*args, **kwargs)

    def get_item(self, index, shift=None):
        img, target, meta = super(CharadesVideoSfm2, self).get_item(index, shift)
        video = img.copy()
        video *= np.array([0.229, 0.224, 0.225])[None, None, None, :]
        video += np.array([0.485, 0.456, 0.406])[None, None, None, :]
        video = np.asarray(np.clip(video*255., 0, 255), dtype="uint8")
        prev_frame = video[0]
        frame = video[1]
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        if mag.mean() < 1:
            raise ValueError('Less then 1 pixel average optical flow in the frame, skipping')
        return img, target, meta
