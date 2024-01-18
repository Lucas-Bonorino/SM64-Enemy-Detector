from os import listdir
from os.path import abspath, isdir, isfile
import cv2 as cv
import numpy as np
from numba import njit
import re

@njit
def change_color(Texture, new_Color):
    Height, Width=Texture.shape[:2]

    for i in range(0, Height):
        for j in range(0, Width):
            if(Texture[i, j, 3]!=0):
                Texture[i,j,:]=new_Color[:]

def change_vertex_color(name, substituto1, substituto2, substituto3):
    
    if not isfile(abspath(f'{name}\\model.inc.c')):
        return

    with open(f'{name}\\model.inc.c', 'r') as arquivo:
        model=arquivo.read()

    padrao1 = r'{0x[0-9A-Fa-f]{2}, 0x[0-9A-Fa-f]{2}, 0x[0-9A-Fa-f]{2}, 0x00}'
    padrao2 = r'{0x[0-9A-Fa-f]{2}, 0x[0-9A-Fa-f]{2}, 0x[0-9A-Fa-f]{2}, 0xff}'
    padrao3 = r'0x[0-9A-Fa-f]{2}, 0x[0-9A-Fa-f]{2}, 0x[0-9A-Fa-f]{2}'

    new_model=re.sub(padrao1, substituto1, model)  
    
    new_model=re.sub(padrao2, substituto2, new_model)
    
    new_model=re.sub(padrao3, substituto3, new_model)

    with open(f'{name}\\model.inc.c', 'w') as arquivo:
        arquivo.write(new_model)


def Color_Correction(directory_list, color):
    s1=f'{{0x{hex(color[2])[2:].zfill(2)}, 0x{hex(color[1])[2:].zfill(2)}, 0x{hex(color[0])[2:].zfill(2)}, 0x{hex(0)[2:].zfill(2)}}}'
    s2=f'{{0x{hex(color[2])[2:].zfill(2)}, 0x{hex(color[1])[2:].zfill(2)}, 0x{hex(color[0])[2:].zfill(2)}, 0x{hex(255)[2:].zfill(2)}}}'
    s3=f'0x{hex(color[2])[2:].zfill(2)}, 0x{hex(color[1])[2:].zfill(2)}, 0x{hex(color[0])[2:].zfill(2)}'

    for directory in directory_list:
        texture_names=listdir(abspath(directory))
       
        change_vertex_color(directory, s1,s2,s3)

        for texture_name in texture_names:

            if 'png' not in texture_name: continue
            
            texture=cv.imread(f'{directory}\\{texture_name}', flags=cv.IMREAD_UNCHANGED)
            change_color(texture, color)
            cv.imwrite(f'{directory}\\{texture_name}', texture)



neutral_directories=[dir for dir in listdir() if isdir(dir)]
neutral_color=[0, 0, 0, 255]

Mario_Dir=['mario', 'mario_cap']
Mario_Color=[0, 255, 0, 255]

Goomba_Dir=['goomba']
Goomba_Color=[114, 128, 250, 255]

Chain_Chomp_Dir=['chain_chomp', 'chain_ball']
Chain_Chomp_Color=[60, 20, 220, 255]

Bobomb_king_Dir=['king_bobomb']
Bobomb_king_Color=[0, 0, 255, 255]

Thwomp_Dir=['thwomp']
Thwomp_Color=[0,0,138, 255]

Whomp_Dir=['whomp']
Whomp_Color=[34,34,178,255]

Piranha_Dir=['piranha_plant']
Piranha_Color=[42,42,165, 255]

directories=[neutral_directories, Mario_Dir, Goomba_Dir, Chain_Chomp_Dir, Bobomb_king_Dir, Thwomp_Dir, Whomp_Dir, Piranha_Dir]
colors=[neutral_color, Mario_Color, Goomba_Color, Chain_Chomp_Color, Bobomb_king_Color, Thwomp_Color, Whomp_Color, Piranha_Color]

directories_and_colors=list(zip(directories, colors))

for directory, color in directories_and_colors:
    Color_Correction(directory, color)