import numpy as np
from numba import njit
import cv2 as cv
from os import makedirs

path='./MarioNet64 groundtruth'

try:
    makedirs(path)
except OSError as error:
    print(error)


ally_color=np.uint8([0, 255, 0, 1])
enemy_color=np.uint8([0, 0, 255, 2])
danger_color=np.uint8([255, 0, 255, 3])
mobility_color=np.uint8([255, 255, 0, 4])
valuable_color=np.uint8([0, 255, 255, 5])


colors=[enemy_color, mobility_color, danger_color, valuable_color, ally_color]
thresholds=[50, 50, 150, 150, 150]

class_colors_and_thresholds=list(zip(colors,thresholds))

nome_da_captura='Compressed Videos\\Segmented.mp4'
video= cv.VideoCapture(nome_da_captura)


@njit
def Is_Similar_Color(color, class_color, threshold):
    norm=0
    for i in range(0,3):
        norm+=(color[i]-class_color[i])**2

    norm=np.sqrt(norm)

    return(norm<threshold)

@njit 
def Is_Black(color):
    black=((color[0]==0) and (color[1]==0) and (color[2]==0))
    return(black)

@njit 
def Video_Process_Pass(frame, class_color, shape, dataset_image, threshold):
    Height, Width = shape[:2]
    
    for i in range(0, Height):
        for j in range(0, Width):
            color=frame[i,j,0:3]

            #if(Is_Similar_Color(color, class_color, threshold) and Is_Black(dataset_image[i,j,0:3])):
                #dataset_image[i,j, 0:3]=class_color[0:3]

            if(Is_Similar_Color(color, class_color, threshold) and dataset_image[i,j]==0):
                dataset_image[i,j]=class_color[3]

i=0
while(video.isOpened()):
    ret, frame=video.read()
    
    if(not ret): 
        break

    shape= frame.shape

    dataset_image=np.zeros(shape[:2], dtype=np.uint8)

    for class_color, threshold in class_colors_and_thresholds:
        Video_Process_Pass(frame, class_color, shape, dataset_image, threshold)

    dataset_image=np.where(dataset_image==0, 5, dataset_image-1)

    cv.imwrite(f'MarioNet64 groundtruth/frame {i}.jpg', dataset_image)
    i+=1

video.release()