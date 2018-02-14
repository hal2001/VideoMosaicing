'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''

from matplotlib.patches import ConnectionPatch
import numpy as np
import utils
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from corner_detector import corner_detector
from anms import anms
import pdb
from feat_match import feat_match
import matplotlib.cm as cm


def feat_desc(img, x, y):
    pad = np.zeros([img.shape[0] + 80, img.shape[1] + 80])

    mu = 0
    sigma = 1  # 0.4
    # gau = np.matrix([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])
    # gau = np.asarray(gau)
    gau = utils.GaussianPDF_2D(mu, sigma, 10, 10)
    # gau = 1 / 159 * gau
    '''dx = np.matrix([1, 0, -1])
    dy = np.matrix([[1], [0], [-1]])
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    Gx = signal.convolve2d(gau, dx, 'same')
    Gy = signal.convolve2d(gau, dy, 'same')'''

    lx = signal.convolve2d(img, gau, 'same')
    ly = signal.convolve2d(img, gau, 'same')

    blurImg = lx + ly
    #plt.imshow(blurImg)
    # plt.show()
    pad[40:img.shape[0] + 40, 40:img.shape[1] + 40] = blurImg
    offset = 40
    output = np.zeros([64, x.size])
    for i in range(x.size):
        windowB = pad[x[i][0] + offset - 20:x[i][0] + offset + 20, y[i][0] + offset - 20:y[i][0] + offset + 20]
        for j in range(0, 64):
            row = (j + 1) // 8
            if (j + 1) % 8 == 0:
                row = (j + 1) // 8 - 1
            col = (j + 1) % 8 - 1
            if col == -1:
                col = 7
            subB = windowB[0 + row*5:row*5 + 4, 0 + col*5:col*5+4]
            output[j, i] = subB.max()
        output_mean = np.sum(output[:, i]) / 64
        output_var = np.std(output[:, i])
        if output_var != 0:
            output[:, i] = (output[:, i] - output_mean) / output_var
        else:
            output[:, i] = 0
    return output


if __name__ == '__main__':
    I = np.array(Image.open('im1.jpg').convert('RGB'))
    im_gray = utils.rgb2gray(I)
    cimg = corner_detector(im_gray)
    Icopy = I.copy()
    Icopy[cimg > 0] = [0, 0, 255]
    plt.subplot(2, 2, 1)
    plt.imshow(Icopy)

    x1, y1, rmax = anms(cimg, 1000)
    Icopyy = np.zeros([np.shape(I)[0], np.shape(I)[1]])
    Icopyy[y1, x1] = 1
    plt.subplot(2, 2, 2)
    plt.imshow(Icopyy)
    # plt.show()
    output1 = feat_desc(im_gray, y1, x1)
    print(output1)

    I2 = np.array(Image.open('im2.jpg').convert('RGB'))
    im_gray2 = utils.rgb2gray(I2)
    cimg2 = corner_detector(im_gray2)
    Icopy2 = I2.copy()
    Icopy2[cimg2 > 0] = [255, 0, 0]
    plt.subplot(2, 2, 3)
    plt.imshow(Icopy2)

    x2, y2, rmax = anms(cimg2, 1000)
    Icopyy2 = np.zeros([I2.shape[0], I2.shape[1]])
    Icopyy2[y2, x2] = 1
    plt.subplot(2, 2, 4)
    plt.imshow(Icopyy2)
    plt.show()
    output2 = feat_desc(im_gray2, y2, x2)
    print(output2)

    match = feat_match(output1, output2)
    a = (match != -1).sum()
    xl = np.zeros([a, 1])
    xr = np.zeros([a, 1])
    yl = np.zeros([a, 1])
    yr = np.zeros([a, 1])
    counter = 0
    for i in range(0, len(match)):
        if match[i] != -1:
            xl[counter] = x1[i]
            yl[counter] = y1[i]
            xr[counter] = x2[match[i]]
            yr[counter] = y2[match[i]]
            counter += 1

    I1 = I.copy()
    # fig1 = plt.figure()
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(I1)
    colors = cm.rainbow(np.linspace(0, 1, len(yl)))
    for x, y, c in zip(xl, yl, colors):
        ax1.scatter(x, y, color=c)

    I2 = I2.copy()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(I2)
    for x, y, c in zip(xr, yr, colors):
        ax2.scatter(x, y, color=c)
    plt.show()
