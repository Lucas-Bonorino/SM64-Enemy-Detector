import numpy as np
from numba import njit
import cv2 as cv
from os import makedirs

SAMPLE_RATE=1

path='./MarioNet64_Segmentation'

try:
    makedirs(path)
except OSError as error:
    print(error)

path='./MarioNet64 Raw Frames'

try:
    makedirs(path)
except OSError as error:
    print(error)

Mario_color=np.uint8([0, 255, 0, 1])
Goomba_color=np.uint8([0, 0, 255, 2])
Chain_Chomp_color=np.uint8([127, 0, 127, 3])
Bobomb_Color=np.uint8([0, 255, 255, 4])
Bobomb_King_Color=np.uint8([255, 0, 0, 5])
Thwomp_Color=np.uint8([127,127,0, 6])
Whomp_Color=np.uint8([0,0,127, 7])
Piranha_Color=np.uint8([255,255,0, 8])

colors=[Mario_color,  Goomba_color,  Chain_Chomp_color, Bobomb_Color, Bobomb_King_Color, Thwomp_Color, Whomp_Color, Piranha_Color]
thresholds=[40, 80, 100, 20, 40, 50, 80, 50]

class_colors_and_thresholds=list(zip(colors,thresholds))

nome_da_captura='Training Videos\\Segmented.mp4'
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

            if(Is_Similar_Color(color, class_color, threshold) and dataset_image[i,j]==0):
                dataset_image[i,j]=class_color[3]

i=0
frame_num=0

while(video.isOpened()):
    ret, frame=video.read()
    
    if(not ret): break

    if(i % SAMPLE_RATE==0):
        frame=cv.resize(frame, (720, 405))
        
        shape= frame.shape

        dataset_image=np.zeros(shape[:2], dtype=np.uint8)

        for class_color, threshold in class_colors_and_thresholds:
            Video_Process_Pass(frame, class_color, shape, dataset_image, threshold)

        dataset_image=np.where(dataset_image==0, 255, dataset_image-1)
     
        cv.imwrite(f'MarioNet64_Segmentation/frame_{frame_num}.jpg', dataset_image)

        frame_num+=1

    i+=1

video.release()
