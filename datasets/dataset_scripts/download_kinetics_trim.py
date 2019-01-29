#!/usr/bin/python
# Download video links with youtube-dl
# Gul Varol and Gunnar Atli Sigurdsson
# 2016
#
# Usage: ./download_kinetics_trim.py outdir/ csvfile.csv

import csv
import sys
import os
import itertools
import numpy
import time
import glob
import subprocess
from random import shuffle

verbose = 1

# Locate youtube-dl binary (edit this if you have it elsewhere)
youtubedl = 'youtubedl/youtube-dl'
if not os.path.exists(youtubedl):
    raise OSError('Cannot find youtube-dl!')

outdir = sys.argv[1]

# Create output directories if not exist
if not os.path.exists(outdir):
    os.mkdir(outdir)

skipped_video_ix = []
cnt = 0
t1  = time.time()
files = sys.argv[2:]
for filename in files:
    with open(filename) as csvfile:
        reader = list(csv.DictReader(csvfile))
    for row in reader:
        try:
            cnt += 1
            # label,youtube_id,time_start,time_end,split
            assignmentid = row['youtube_id']
            actions = row['label'].replace(' ', '_')
            time_start = row['time_start']
            time_end = row['time_end']
            videolink = "https://www.youtube.com/watch?v="+assignmentid
        
            out_video_dir = '{}/{}'.format(outdir, actions)
            if not os.path.exists(out_video_dir):
                os.mkdir(out_video_dir)
            out_video = '{}/{}_{:06d}_{:06d}.mp4'.format(out_video_dir, assignmentid, time_start, time_end)
            cmd_link = "%s -g \"%s\"" % (youtubedl, videolink)
        
            #### Check if the video is already downloaded
            fileFound = glob.glob("%s.*" % out_video)
            if fileFound:
                if verbose:
                    print "%d: Video already downloaded! (%s)" %(cnt, assignmentid)
            else:
                # Download video
                out = subprocess.check_output(cmd_link, shell=True)
                path = out.split('\n')[0]
                cmd_part = 'ffmpeg -i "{}" -ss {} -to {} -c copy {}'.format(path, time_start, time_end, out_video)
                os.system(cmd_part)

            # Check if the video is downloaded now
            fileFound = glob.glob("%s.*" % out_video)
            if not fileFound:
                print "%d: Video skipped! (%s)(%s)" %(cnt, assignmentid, videolink)
                skipped_video_ix += [cnt]
            else:
                if verbose:
                    print "%d: Video downloaded. (%s)" % (cnt, assignmentid)
            
        except Exception as err:
            print(err)
    
        
print "Skipped video indices:"
print skipped_video_ix
elapsed = time.time() - t1
print "%f seconds for %d videos" % (elapsed, cnt)

