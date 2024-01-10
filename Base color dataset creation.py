import numpy as np
from numba import njit
import cv2 as cv
from os import makedirs
from random import randint

SAMPLE_RATE=30

path='./MarioNet64 Color'

try:
    makedirs(path)
except OSError as error:
    print(error)

nome_da_captura='Compressed Videos\\Unsegmented.mp4'
video= cv.VideoCapture(nome_da_captura)

i=0
while(video.isOpened()):
    ret, frame=video.read()

    if not ret: break

    cv.imwrite(f'MarioNet64 Color/frame {i}.jpg', frame)
 
    i+=1


        
