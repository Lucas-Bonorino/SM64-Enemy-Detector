import numpy as np
import cv2 as cv
from os.path import abspath, getctime
from os import makedirs, listdir
from re import sub
import pandas as pd
import sys
from math import floor

ACCEPTABLE_DETECTION_LOWER_THRESHOLD=500
ACCEPTABLE_DETECTION_HIGHER_THRESHOLD=700 * 400
ACCEPTABLE_HEIGHT_WIDTH_RATION=0.2
MAX_CLASSES = 8

    
def filter_marios(BBoxes):
    mario_BBoxes=BBoxes

    if len(mario_BBoxes)<=1: return(mario_BBoxes)

    areas=[]
    for BBOX in mario_BBoxes:
        rec=cv.boundingRect(BBOX)
        area=rec[2]*rec[3]
        areas.append(area)

    mario_real_bbox=mario_BBoxes[np.argmax(areas)]

    return(mario_BBoxes)

def Initialize_data_frame():
    data={'filename':[], 'bounding_boxes':[]}

    return(pd.DataFrame(data))


def Create_DataFrame_Row(bboxes, image_name, classes):
   
    BBOX_Data={'boxes':bboxes, 'classes':classes}
    data={'filename':image_name, 'bounding_boxes': BBOX_Data}
    
    return(pd.DataFrame(data))


def Calculate_BBoxes(segmented_image, image_name, bbox_data):
    Boxes=[]
    Labels=[]
    for class_number in range(MAX_CLASSES):

        binary_image=np.where(segmented_image==class_number, 255, 0).astype(np.uint8)

        c,h=cv.findContours(binary_image, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)

        if class_number==0:
            c=filter_marios(c)

        for contour in c:
            bounding_rectangle=cv.boundingRect(contour)

            height=bounding_rectangle[2]
            width=bounding_rectangle[3]

            if(width * height < ACCEPTABLE_DETECTION_LOWER_THRESHOLD) or (height * width > ACCEPTABLE_DETECTION_HIGHER_THRESHOLD): continue

            if(height/width<ACCEPTABLE_HEIGHT_WIDTH_RATION) or (width/height<ACCEPTABLE_HEIGHT_WIDTH_RATION): continue



            Boxes.append(bounding_rectangle)
            Labels.append(class_number)

    if(len(Labels)>=1):
        row=Create_DataFrame_Row(Boxes, image_name, Labels)
        bbox_data=pd.concat([bbox_data, row], ignore_index=True)

    return(bbox_data)

def Criteria(item):
    num_string=sub('[^0-9]', '', item)
    return(int(num_string))

def Progres_bar(i, num):
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    sys.stdout.flush()
    progress=(i/num)
    Bar_Progress=floor(progress*30)
    Remaining_Bar=30-Bar_Progress
    print('['+Bar_Progress*'='+' '*Remaining_Bar+']'+str(100*progress)+'% concluido',end='', flush=True)

def Create_Dataset(segmented_folder_name, unsegmented_folder_name):
    segmented_images = sorted(listdir(abspath(segmented_folder_name)), key=Criteria)
    
    unsegmented_images = sorted(listdir(abspath(unsegmented_folder_name)), key=Criteria)

    images=list(zip(segmented_images, unsegmented_images))

    BBox_list=Initialize_data_frame()
    num=len(images)
    i=0
    for segmented_image_name, unsegmented_image_name in images:
        segmented_image_name=segmented_folder_name+'\\'+segmented_image_name
        color_image_name=unsegmented_folder_name+'\\'+unsegmented_image_name
        
        segmented_image = cv.imread(segmented_image_name, cv.COLOR_BGR2GRAY)

        BBox_list = Calculate_BBoxes(segmented_image, color_image_name, BBox_list)
        i+=1
        Progres_bar(i,num)
        
    BBox_list.to_csv("../Bounding Box Annotations ex.csv")
        
if __name__=='__main__':
    Create_Dataset('..\\MarioNet64_segmentation','..\\MarioNet64_Color')


