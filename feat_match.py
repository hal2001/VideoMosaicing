# -*- coding: utf-8 -*-
'''
  File name: feat_match.py
  Author: Hantian Liu
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''
import numpy as np
import math

def feat_match(descs1, descs2):
  n1=np.shape(descs1)[1]
  n2=np.shape(descs2)[1]
  dict={}
  match=np.zeros([n1,1])
  '''
  for i in range(0, n1):
    for j in range(0, n2):
      dict[j]=np.square(descs1[:,i]-descs2[:,j]).sum()
  '''
  '''
  sorted_list = sorted(dict.items(), key = lambda x: x[1]) #smaller distance to larger distance
  if sorted_list[0][1]<0.6*sorted_list[1][1]:
    match[i]=sorted_list[0][0]
  else:      
    match[i]=-1
  '''
  for k in range(0, n1):
      descs1_k=np.array([descs1[:,k]])
      descs1_k=descs1_k.transpose()
      diff=descs2-descs1_k
      diff=np.square(diff)
      s=diff.sum(axis=0)
      ind=np.argmin(s)
      #ind=np.where(s==min(s))
      #ind=ind[0][0]
      min1=s[ind]
      s[ind]=math.inf
      ind2=np.argmin(s)
      #ind2=np.where(s==min(s))
      #ind2=ind2[0][0]

      if min1<0.6*s[ind2]:
          if any(match==ind):
              match[k]=-1
          else:
              match[k]=ind
      else:
          match[k]=-1

  match=match.astype('int64')
  return match
