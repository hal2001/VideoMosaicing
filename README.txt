Run test_mosaic.py to start!

Details:
1. input videos are in video folder
2. every 5 frames, we extract a frame to do mosaic
3. in mymosaic.py, pts_num is the expected number of points after nms, threshold is the threshold to determine the inliers of ransac.
4. result video is mosaic.avi
5. plt.rcParams['animation.ffmpeg_path'] should be the path of your ffmpeg.exe