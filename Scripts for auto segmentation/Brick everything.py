from os import listdir
from os.path import abspath, isdir
import cv2 as cv
import numpy as np
from numba import njit

@njit
def change_color(Texture, new_Color):
    Height, Width=Texture.shape[:2]
    noalpha=False

    if(Texture.shape[2]<4):
        noalpha=True

    for i in range(0, Height):
        for j in range(0, Width):
            if(noalpha or Texture[i, j, 3]!=0):
                Texture[i,j,0:Texture.shape[2]]=new_Color[0:Texture.shape[2]]

def Color_Correction(directory_list, color):
    for directory in directory_list:
        texture_names=listdir(abspath(directory))

        for texture_name in texture_names:
            texture=cv.imread(f'{directory}\\{texture_name}', flags=cv.IMREAD_UNCHANGED)
            change_color(texture, color)
            cv.imwrite(f'{directory}\\{texture_name}', texture)

neutral_color=[0, 0, 0, 255]

Color_Correction([ f for f in listdir() if isdir(f)], neutral_color)