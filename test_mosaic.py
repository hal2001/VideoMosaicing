from mymosaic import mymosaic
import numpy as np
import imageio
import os
from myvideomosaic import myvideomosaic
import matplotlib.pyplot as plt

img_input = []
folder = 'video'
numV = 0
length = []
lastone = np.array([0])

# find the max length of videos
for filename in os.listdir(folder):
    im_path = os.path.join(folder, filename)
    reader = imageio.get_reader(im_path)
    length.append(len(reader))
maxi = max(length)

# get img_input
for filename in os.listdir(folder):
    # read in video
    im_path = os.path.join(folder, filename)
    reader = imageio.get_reader(im_path)

    # if it is the first video
    if numV == 0:
        for i, im in enumerate(reader):
            img_input.append([im])
            lastone = im
        lennow = len(reader)
        while lennow < maxi:
            img_input.append([lastone])
            lennow = lennow + 1
        numV = numV + 1

    # next videos
    else:
        for i, im in enumerate(reader):
            img_input[i].append(im)
            lastone = im
        lennow = len(reader)
        while lennow < maxi:
            img_input[lennow].append(lastone)
            lennow = lennow + 1

img_input1 = []
for i in range(0, len(img_input) // 10):
    img_input1.append(img_input[i*10])
img_mosaic = mymosaic(img_input1)
myvideomosaic(img_mosaic)

