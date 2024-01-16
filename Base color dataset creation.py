import numpy as np
from numba import njit
import cv2 as cv
from os import makedirs
from random import randint

SAMPLE_RATE=5

path='./MarioNet64 Object Detector Training Dataset'

try:
    makedirs(path)
except OSError as error:
    print(error)

nome_da_captura='Training Videos\\Unsegmented.mp4'
video= cv.VideoCapture(nome_da_captura)

i=0
frame_num=0
while(video.isOpened()):
    ret, frame=video.read()

    if not ret: break

    if (i % SAMPLE_RATE==0):
        frame=cv.resize(frame,(720, 405))
        cv.imwrite(f'MarioNet64 Object Detector Training Dataset/frame {frame_num}.jpg', frame)
        frame_num+=1
 
    i+=1


        
