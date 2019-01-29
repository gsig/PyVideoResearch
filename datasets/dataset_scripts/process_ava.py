import subprocess
import os
import numpy as np
import glob

timestamps = range(902,1798+1)
clip_size = 3

outdir = 'processed_videos2/'
videos = glob.glob('videos/*.mp4')

for path in videos:
    for time_start in np.arange(902-clip_size, 1798+clip_size+1, clip_size):
        print('{} {}'.format(path, time_start))
        base, ext = os.path.splitext(path)
        base = base.split('/')[-1]
        time_end = time_start+clip_size
        out_video = '{}{}_{}_{}{}'.format(outdir, base, int(time_start), int(time_end), ext)
        cmd_part = 'ffmpeg -i "{}" -ss {} -to {} -n -c:v libx264 -c:a copy {}'.format(path, time_start, time_end, out_video)
        os.system(cmd_part)
