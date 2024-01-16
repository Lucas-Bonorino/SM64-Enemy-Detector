import numpy as np
import cv2 as cv
from os.path import abspath, getctime
from os import makedirs, listdir
from re import sub

ACCEPTABLE_DETECTION_THRESHOLD=40
ally_color=(0, 255, 0)
enemy_color=(0, 0, 255)
danger_color=(255, 0, 255)
colors=[ally_color, enemy_color, danger_color]

path='./MarioNet64 groundtruth'

try:
    makedirs(path)
except OSError as error:
    print(error)

def Calculate_BBOX(connected_components, label, class_color, color_image):
    component_positions=np.column_stack(np.where(connected_components == label))
        
    minx=np.min(component_positions[:, 0])
    miny=np.min(component_positions[:, 1])
    startpoint=(miny, minx)

    maxx=np.max(component_positions[:, 0])
    maxy=np.max(component_positions[:, 1])
    endpoint=(maxy, maxx)

    if((maxx-minx) * (maxy-miny) < ACCEPTABLE_DETECTION_THRESHOLD): return(color_image)

    color_image=cv.rectangle(color_image, startpoint, endpoint, class_color)

    return(color_image)

def Calculate_GD(segmented_image, unsegmented_image):
    Ground_Truth=unsegmented_image
   
    for class_number in range(len(colors)):

        binary_image=np.where(segmented_image==class_number, 255, 0).astype(np.uint8)

        label_num, connected_components=cv.connectedComponents(binary_image)

        for label in range(label_num):
            Ground_Truth=Calculate_BBOX(connected_components, label, colors[class_number], Ground_Truth)

    return(Ground_Truth)

def Criteria(item):
    num_string=sub('[^0-9]', '', item)
    return(int(num_string))


def Create_Dataset(segmented_folder_name, unsegmented_folder_name):

    segmented_images = sorted(listdir(abspath(segmented_folder_name)), key=Criteria)
    
    unsegmented_images = sorted(listdir(abspath(unsegmented_folder_name)), key=Criteria)

    images=list(zip(segmented_images, unsegmented_images))

    i=0
    for segmented_image_name, unsegmented_image_name in images:
        segmented_image=cv.imread(segmented_folder_name+'\\'+segmented_image_name, cv.COLOR_BGR2GRAY)

        unsegmented_image=cv.imread(unsegmented_folder_name+'\\'+unsegmented_image_name)

        GD=Calculate_GD(segmented_image, unsegmented_image)

        cv.imwrite(f'MarioNet64 groundtruth/frame {i}.jpg', GD)
        i+=1

if __name__=='__main__':
    Create_Dataset('MarioNet64 segmetation','MarioNet64 Base Color')


