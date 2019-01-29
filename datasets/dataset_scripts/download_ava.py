#!/usr/bin/python
# Download video links for AVA with youtube-dl
# Gul Varol and Gunnar Atli Sigurdsson
# 2016
#
# Usage: ./download_ava.py outdir/ csvfile.csv 

import csv
import sys
import os
import itertools
import numpy
import time
import glob
from random import shuffle

verbose = 0

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
shuffle(files)
for filename in files:
    with open(filename) as csvfile:
        reader = list(csv.reader(csvfile))
        shuffle(reader)
    for row in reader:
        print(row)
        cnt += 1
        # label,youtube_id,time_start,time_end,split
        assignmentid = row[0]
        videolink = "https://www.youtube.com/watch?v="+assignmentid
    
        out_video = os.path.join(outdir, assignmentid)
        
        cmd_download = "%s \"%s\" -ci --min-filesize 10k -o \"%s.%s\"" % (youtubedl, videolink, out_video, '%(ext)s')
    
        #### Check if the video is already downloaded
        fileFound = glob.glob("%s.*" % out_video)
        if fileFound:
            if verbose:
                print "%d: Video already downloaded! (%s)" %(cnt, assignmentid)
        else:
            # Download video
            os.system(cmd_download)

        # Check if the video is downloaded now
        fileFound = glob.glob("%s.*" % out_video)
        if not fileFound:
            print "%d: Video skipped! (%s)(%s)" %(cnt, assignmentid, videolink)
            skipped_video_ix += [cnt]
        else:
            if verbose:
                print "%d: Video downloaded. (%s)" % (cnt, assignmentid)
        
print "Skipped video indices:"
print skipped_video_ix

elapsed = time.time() - t1
print "%f seconds for %d videos" % (elapsed, cnt)

