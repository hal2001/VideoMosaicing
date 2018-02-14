# -*- coding: utf-8 -*-
'''
  File name: anms.py
  Author: Hantian Liu
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

import numpy as np
import math

def anms(cimg, max_pts):
  ind=np.where(cimg!=0)
  y_corner=ind[0] #row
  x_corner=ind[1] #col
  dict = {}

  for i in range(0,len(x_corner)):
    himg=cimg.copy()
    himg[y_corner[i],x_corner[i]]=0;
    ind_large=np.where(himg>0.9*cimg[y_corner[i],x_corner[i]])
    if len(ind_large[0])==0:
        dict[(x_corner[i], y_corner[i])] = math.inf
        continue
    y_large=ind_large[0]
    x_large=ind_large[1]
    dis=np.square(x_large-x_corner[i])+np.square(y_large-y_corner[i])
    radius=dis.min()
    dict[(x_corner[i], y_corner[i])] = radius

  sorted_list=sorted(dict.items(), key = lambda x: x[1], reverse=True) #large radius to small radius
  #while sorted_list[max_pts-1]==sorted_list[max_pts]:
  #  max_pts=max_pts-1

  want_list = sorted_list[0:max_pts]
  rmax=want_list[-1][1]
  x=np.zeros([len(want_list),1])
  y = np.zeros([len(want_list),1])
  for i in range(0,len(want_list)):
    x[i]=want_list[i][0][0]
    y[i]=want_list[i][0][1]
  x=x.astype('int64')
  y=y.astype('int64')
  return x, y, rmax
