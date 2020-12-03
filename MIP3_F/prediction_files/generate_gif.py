import matplotlib.pyplot as plt 
import cv2
import os
import pdb
import glob
import numpy as np
import re

video_name = 'video.avi'

images = glob.glob('./images/image_*.png')
images=sorted(images, key= lambda x: (np.array([re.findall("(\d+)",x)]).astype(int)*np.array([16*16, 16, 1])).sum())
frame = cv2.imread( images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, fps=30, frameSize=(width//8,height//8))
for i,image in enumerate(images):
    print('{}/{}'.format(i,len(images)))
    video.write(cv2.resize(cv2.imread( image),(width//8,height//8)))

cv2.destroyAllWindows()
video.release()