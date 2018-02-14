# -*- coding: utf-8 -*-
'''
  File name: ransac_est_homography.py
  Author: Hantian Liu
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import numpy as np
import random
from est_homography import est_homography
from est_homography import est_homography2
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def EstimateHomography(x_to,y_to,X_from,Y_from):
    A = np.zeros([X_from.size*2,9])
    for i in range(0,X_from.size):
      a = np.matrix(np.array([X_from[i],Y_from[i],1]))
      b = np.zeros([1,3])
      c = np.array([[x_to[i]], [y_to[i]]])
      d = np.dot(-c,a)
      A[i*2, 0:3]=a
      A[i*2, 3:6]=b
      A[i*2+1,0:3]=b
      A[i * 2 + 1,3:6] = a
      A[i*2:i*2+2,6:9] = d

    U, S, V = np.linalg.svd(A, full_matrices=True)
    h = V[:,-1]
    H = h.reshape([3,3])
    H=H.transpose()
    H = H/H[2, 2] # for calibration!
    return H

def ransac_est_homography(x1, y1, x2, y2, thresh): #, I1, I2): # from 1 to 2
    nRANSAC=4000
    n=len(x1)
    max_inlier_num=0
    for repeat in range(0, nRANSAC):
        ind = np.random.randint(n - 1, size=4)
        #H=EstimateHomography(x2[ind],y2[ind],x1[ind],y1[ind])
        x11=x1[ind]
        y11=y1[ind]
        x22=x2[ind]
        y22=y2[ind]
        H = est_homography(x11, y11, x22, y22)
        inlier_num=0
        inlier_ind=np.zeros([n,1])
        #fig = plt.figure()
        #colors = cm.rainbow(np.linspace(0, 1, n)) #for c,i in zip(colors,range(0, n)):
        for i in range(0,n):
            #if i==ind[0] or i==ind[1] or i==ind[2] or i==ind[3]:
            #	continue
            pos1=np.array([[x1[i]],[y1[i]],[1]])
            pos1in2=np.dot(H,pos1)
            x1in2=pos1in2[0]/pos1in2[2]
            y1in2=pos1in2[1]/pos1in2[2]
            err=abs(x1in2-x2[i])+abs(y1in2-y2[i])
            err=err[0][0]
            if err<thresh: #inlier
                inlier_num=inlier_num+1
                inlier_ind[i]=1
            else: #outlier
                inlier_ind[i]=0

            #ax1 = fig.add_subplot(1, 2, 1)
            #ax1.imshow(I1)
            #ax1.scatter(x1[i], y1[i], color = c)
            #ax2 = fig.add_subplot(1, 2, 2)
            #ax2.imshow(I2)
            #ax2.scatter(x1in2, y1in2, color = c)
            #plt.show()

        if inlier_num>max_inlier_num:
            max_inlier_num=inlier_num
            max_inlier_ind=inlier_ind
            #max_H=H
    x1in=x1[max_inlier_ind!=0]
    y1in=y1[max_inlier_ind!=0]
    x2in=x2[max_inlier_ind!=0]
    y2in=y2[max_inlier_ind!=0]
    max_H = est_homography2(x1in, y1in, x2in, y2in)
    '''
    g=len(x1in)
    ptstest = np.array([[x1in[0]], [y1in[0]], [1]])
    print(ptstest)
    print('left in center')
    result = np.dot(max_H, ptstest)
    result[0, :] = result[0, :] / result[2, :]
    result[1, :] = result[1, :] / result[2, :]
    print(result)
    centertest = np.array([[x2in[0]], [y2in[0]], [1]])
    print('center')
    '''
    return max_H, max_inlier_ind
