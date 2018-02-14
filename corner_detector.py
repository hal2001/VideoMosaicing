# -*- coding: utf-8 -*-
'''
  File name: corner_detector.py
  Author: Hantian Liu
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''
import utils
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from anms import anms
from skimage.feature import corner_harris


def corner_detector(img):
    cimg = np.zeros([img.shape[0], img.shape[1]])
    img = img.astype(np.float32)
    cimg = cv2.cornerHarris(img, 2, 3, 0.04)
    #cimg = corner_harris(img)
    cimg[cimg < 0.1 * cimg.max()] = 0
    print('detected points:')
    print(cimg[cimg != 0].size)
    return cimg


if __name__ == '__main__':
    I = np.array(Image.open('im1.jpg').convert('RGB'))
    im_gray = utils.rgb2gray(I)
    cimg = corner_detector(im_gray)
    Icopy = I.copy()
    Icopy[cimg > 0] = [0, 0, 255]
    plt.subplot(1, 2, 1)
    plt.imshow(Icopy)
    features = cimg[cimg > 0].size

    x, y, rmax = anms(cimg, round(features * 0.1))
    print(x.size)
    Icopyy = np.zeros([np.shape(I)[0], np.shape(I)[1]])
    Icopyy[y, x] = 1
    plt.subplot(1, 2, 2)
    plt.imshow(Icopyy)
    plt.show()
