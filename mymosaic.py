#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/11/17 4:45 PM 

@author: Hantian Liu
"""

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''

import numpy as np
import utils
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from corner_detector import corner_detector
from anms import anms
import pdb
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from feat_desc import feat_desc
from scipy import ndimage
from ransac_est_homography import EstimateHomography
from est_homography import est_homography
import matplotlib.cm as cm
import cv2

pts_num = 700  # TODO
thresh = 4  # TODO
pad = 0  # TODO


def MatchToH(match, xc, yc, xadd, yadd, middle, left): #x for column & y for row
    match_num = (match != -1).sum()
    x_c = np.zeros([match_num, 1])
    y_c = np.zeros([match_num, 1])
    x = np.zeros([match_num, 1])
    y = np.zeros([match_num, 1])
    counter = 0
    for i in range(0, len(match)):
        if match[i] != -1:
            x_c[counter] = xc[i]
            y_c[counter] = yc[i]
            x[counter] = xadd[match[i]]
            y[counter] = yadd[match[i]]
            counter += 1
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(middle)
    colors = cm.rainbow(np.linspace(0, 1, len(x_c)))
    for x0, y0, c in zip(x_c, y_c, colors):
        ax1.scatter(x0, y0, color = c)

    ax2 = fig.add_subplot(122)
    ax2.imshow(left)
    for x0, y0, c in zip(x, y, colors):
        ax2.scatter(x0, y0, color = c)

    plt.show()
    '''
    H, inlier_ind = ransac_est_homography(x, y, x_c, y_c, thresh)
    return H, inlier_ind


def findCanvas(H, center, left):
    '''
    # H*[row;col;1]
    h, w, z = left.shape
    hm, wm, zm = center.shape
    ver1 = np.array([[0], [0], [1]])
    ver2 = np.array([[h - 1], [0], [1]])
    ver3 = np.array([[0], [w - 1], [1]])
    ver4 = np.array([[h - 1], [w - 1], [1]])
    ver_in_c = np.zeros([4, 3])
    ver_in_c[0, :] = (np.dot(H, ver1)).transpose()
    ver_in_c[0, :] = ver_in_c[0, :] / ver_in_c[0, 2]
    ver_in_c[1, :] = (np.dot(H, ver2)).transpose()
    ver_in_c[1, :] = ver_in_c[1, :] / ver_in_c[1, 2]
    ver_in_c[2, :] = (np.dot(H, ver3)).transpose()
    ver_in_c[2, :] = ver_in_c[2, :] / ver_in_c[2, 2]
    ver_in_c[3, :] = (np.dot(H, ver4)).transpose()
    ver_in_c[3, :] = ver_in_c[3, :] / ver_in_c[3, 2]

    offsetw = (ver_in_c.min(0)[1] < 0) * (-ver_in_c.min(0)[1])
    offseth = (ver_in_c.min(0)[0] < 0) * (-ver_in_c.min(0)[0])
    bottomw = (ver_in_c.max(0)[1] > wm) * (ver_in_c.max(0)[1] - wm)
    bottomh = (ver_in_c.max(0)[0] > hm) * (ver_in_c.max(0)[0] - hm)
    '''
    # H*[col;row;1]
    h, w, z = left.shape
    hm, wm, zm = center.shape
    ver1 = np.array([[0], [0], [1]])
    ver2 = np.array([[w - 1], [0], [1]])
    ver3 = np.array([[0], [h - 1], [1]])
    ver4 = np.array([[w - 1], [h - 1], [1]])
    ver_in_c = np.zeros([4, 3])
    #print(ver1)
    #print(np.dot(H, ver1))
    #print(ver2)
    #print(np.dot(H, ver2))
    #print(ver3)
    #print(np.dot(H, ver3))
    #print(ver4)
    #print(np.dot(H, ver4))
    ver_in_c[0, :] = (np.dot(H, ver1)).transpose()
    ver_in_c[0, :] = ver_in_c[0, :] / ver_in_c[0, 2]
    ver_in_c[1, :] = (np.dot(H, ver2)).transpose()
    ver_in_c[1, :] = ver_in_c[1, :] / ver_in_c[1, 2]
    ver_in_c[2, :] = (np.dot(H, ver3)).transpose()
    ver_in_c[2, :] = ver_in_c[2, :] / ver_in_c[2, 2]
    ver_in_c[3, :] = (np.dot(H, ver4)).transpose()
    ver_in_c[3, :] = ver_in_c[3, :] / ver_in_c[3, 2]

    offseth = (ver_in_c.min(0)[1] < 0) * (-ver_in_c.min(0)[1])
    offsetw = (ver_in_c.min(0)[0] < 0) * (-ver_in_c.min(0)[0])
    bottomh = (ver_in_c.max(0)[1] > hm) * (ver_in_c.max(0)[1] - hm)
    bottomw = (ver_in_c.max(0)[0] > wm) * (ver_in_c.max(0)[0] - wm)


    new_h = offseth + hm + bottomh
    new_w = offsetw + wm + bottomw

    return int(new_h), int(new_w), int(offseth), int(offsetw)


def Stitch(middle, left):
    h, w, z = left.shape
    center_img = utils.rgb2gray(middle)
    cimg_c = corner_detector(center_img)
    xc, yc, rmaxc = anms(cimg_c, pts_num) # col, row
    output_c = feat_desc(center_img, yc, xc)

    left_img = utils.rgb2gray(left)
    cimg_left = corner_detector(left_img)
    xleft, yleft, rmaxleft = anms(cimg_left, pts_num) # col, row
    output_left = feat_desc(left_img, yleft, xleft)

    match = feat_match(output_c, output_left)
    a = (match != -1).sum()
    if a<20:
        return 0
    '''
    fig = plt.figure()
    axis1 = fig.add_subplot(221)
    axis1.imshow(middle)
    axis1.scatter(xc, yc)
    axis2 = fig.add_subplot(222)
    axis2.imshow(left)
    axis2.scatter(xleft, yleft)

    a = (match != -1).sum()
    xl = np.zeros([a, 1])
    xr = np.zeros([a, 1])
    yl = np.zeros([a, 1])
    yr = np.zeros([a, 1])
    counter = 0
    for i in range(0, len(match)):
        if match[i] != -1:
            xl[counter] = xc[i]
            yl[counter] = yc[i]
            xr[counter] = xleft[match[i]]
            yr[counter] = yleft[match[i]]
            counter += 1

    ax1 = fig.add_subplot(223)
    ax1.imshow(middle)
    colors = cm.rainbow(np.linspace(0, 1, len(yl)))
    for x, y, c in zip(xl, yl, colors):
        ax1.scatter(x, y, color = c)

    ax2 = fig.add_subplot(224)
    ax2.imshow(left)
    for x, y, c in zip(xr, yr, colors):
        ax2.scatter(x, y, color = c)

    plt.show()
    '''

    H, inlier_ind = MatchToH(match, xc, yc, xleft, yleft, middle, left)
    new_h, new_w, offseth, offsetw = findCanvas(H, middle, left)

    x = np.linspace(0, h - 1, h)
    y = np.linspace(0, w - 1, w)
    xv, yv = np.meshgrid(y, x)
    coord_ori = np.ones([3, h * w])
    coord_ori[0, :] = xv.flatten()
    coord_ori[1, :] = yv.flatten()
    coord_ori = coord_ori.astype(np.int)

    coord_new = np.dot(H, coord_ori)
    coord_new = coord_new / coord_new[2, :]
    coord_new[0, :] = coord_new[0, :] + offsetw
    coord_new[1, :] = coord_new[1, :] + offseth
    coord_new = coord_new.astype(np.int)
    # insert left color into new canvas
    new = np.zeros([new_h+1, new_w+1, 3])
    counter=0
    while coord_new.max(axis=1)[0]>new_w or coord_new.max(axis=1)[1]>new_h or coord_new.max(axis=1)[0]<0 or coord_new.max(axis=1)[1]<0:
        print('ayamaya')
        counter=counter+1
        num=200
        cimg_c = corner_detector(center_img)
        xc, yc, rmaxc = anms(cimg_c, pts_num+counter*num)  # col, row
        output_c = feat_desc(center_img, yc, xc)

        cimg_left = corner_detector(left_img)
        xleft, yleft, rmaxleft = anms(cimg_left, pts_num+counter*num)  # col, row
        output_left = feat_desc(left_img, yleft, xleft)

        match = feat_match(output_c, output_left)
        H, inlier_ind = MatchToH(match, xc, yc, xleft, yleft, middle, left)

        fig2=plt.figure()
        ind = np.where(inlier_ind != 0)
        xinl = xc[ind[0]]
        yinl = yc[ind[0]]
        xinr = xleft[ind[0]]
        yinr = yleft[ind[0]]
        ind_out = np.where(inlier_ind == 0)
        xoutl = xc[ind_out[0]]
        youtl = yc[ind_out[0]]
        xoutr = xleft[ind_out[0]]
        youtr = yleft[ind_out[0]]
        ax1 = fig2.add_subplot(223)
        ax1.imshow(center_img)
        ax1.scatter(xinl, yinl, color = 'r')
        ax1.scatter(xoutl, youtl, color = 'b')
        ax2 = fig2.add_subplot(224)
        ax2.imshow(left_img)
        ax2.scatter(xinr, yinr, color = 'r')
        ax2.scatter(xoutr, youtr, color = 'b')
        plt.show()

        new_h, new_w, offseth, offsetw = findCanvas(H, middle, left)
        coord_new = np.dot(H, coord_ori)
        coord_new = coord_new / coord_new[2, :]
        coord_new[0, :] = coord_new[0, :] + offsetw
        coord_new[1, :] = coord_new[1, :] + offseth
        coord_new = coord_new.astype(np.int)
        new = np.zeros([new_h + 1, new_w + 1, 3])

    new[coord_new[1, :], coord_new[0, :], 0] = left[coord_ori[1, :], coord_ori[0, :], 0]
    new[coord_new[1, :], coord_new[0, :], 1] = left[coord_ori[1, :], coord_ori[0, :], 1]
    new[coord_new[1, :], coord_new[0, :], 2] = left[coord_ori[1, :], coord_ori[0, :], 2]

    kernel=np.ones((5,5),np.uint8)
    new=cv2.dilate(new, kernel, iterations=1)

    #new[coord_new[0, :], coord_new[1, :], 0] = left[coord_ori[0, :], coord_ori[1, :], 0]
    #new[coord_new[0, :], coord_new[1, :], 1] = left[coord_ori[0, :], coord_ori[1, :], 1]
    #new[coord_new[0, :], coord_new[1, :], 2] = left[coord_ori[0, :], coord_ori[1, :], 2]

    new_center = np.zeros([new_h+1, new_w+1, 3])
    hm, wm,z=middle.shape
    #plt.figure()
    #plt.imshow(new)
    #plt.show()

    new_center[offseth:offseth + hm, offsetw:offsetw + wm, 0] = middle[:, :, 0]
    new_center[offseth:offseth + hm, offsetw:offsetw + wm, 1] = middle[:, :, 1]
    new_center[offseth:offseth + hm, offsetw:offsetw + wm, 2] = middle[:, :, 2]
    #plt.figure()
    #plt.imshow(new_center)
    #plt.show()
    # insert center color into new canvas
    A = new[offseth:offseth + hm, offsetw:offsetw + wm, :]
    Ac=new_center[offseth:offseth + hm, offsetw:offsetw + wm, :]
    A[np.bitwise_and(A != 0, Ac!=0)] = A[np.bitwise_and(A != 0, Ac!=0)] * 0.5 + Ac[np.bitwise_and(A != 0, Ac!=0)] * 0.5
    #plt.figure()
    #plt.imshow(new)
    #plt.show()

    A[A == 0] = Ac[A == 0]
    #plt.figure()
    #plt.imshow(new)
    #plt.show()
    new = new.astype('uint8')
    return new


def mymosaic(img_input):
    m = len(img_input)  # list includes m sub_list(frame), each sub_list includes n video's m_th frame
    n = len(img_input[0])

    mid = np.floor(n / 2)

    img_mosaic = []
    for frame in range(0, m):
        print('frame')
        print(frame)
        mid = mid.astype('int64')
        im_source = img_input[frame][mid]
        if mid == n / 2:
            im_source = Stitch(im_source, img_input[frame][mid - 1])
            for i in range(0, mid - 1):
                if np.size(im_source) != 1:
                    im_source = Stitch(im_source, img_input[frame][mid - 2 - i])
                if np.size(im_source)!=1:
                    im_source = Stitch(im_source, img_input[frame][mid + 1 + i])
        else:
            for i in range(0, mid):
                im_source = Stitch(im_source, img_input[frame][mid - 1 - i])
                if np.size(im_source) != 1:
                    im_source = Stitch(im_source, img_input[frame][mid + 1 + i])
        if np.size(im_source)!=1:
            img_mosaic.append(im_source)
            plt.imshow(im_source)
            plt.savefig("%s.jpg" %frame)
            #plt.show()
    return img_mosaic
