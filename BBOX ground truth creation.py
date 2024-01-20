import numpy as np
import cv2 as cv
from os.path import abspath, getctime
from os import makedirs, listdir
from re import sub

ACCEPTABLE_DETECTION_LOWER_THRESHOLD=100
ACCEPTABLE_DETECTION_HIGHER_THRESHOLD=700 * 400
ACCEPTABLE_HEIGHT_WIDTH_RATION=0.2
Mario_color=[0, 255, 0]
Goomba_color=[114, 128, 250]
Chain_Chomp_color=[60, 20, 220]
Bobomb_Color=[128, 128, 240]
Bobomb_King_Color=[0, 0, 255]
Thwomp_Color=[0, 0, 138]
Whomp_Color=[34, 34, 178]
Piranha_Color=[42, 42, 165]
colors=[Mario_color,  Goomba_color,  Chain_Chomp_color, Bobomb_Color, Bobomb_King_Color, Thwomp_Color, Whomp_Color, Piranha_Color]
Training_Format_Files=['Mario', 'Goomba', 'Chain Chomp', 'Bobomb', 'Bobomb King', 'Twhomp', 'Whomp', 'Piranha']

path='./MarioNet64 Training'

try:
    makedirs(path)
except OSError as error:
    print(error)


def filter_marios(positives):
    mario_BBoxes=positives[0]

    if len(mario_BBoxes)<=1: return(mario_BBoxes)

    areas=[]
    for BBOX in mario_BBoxes:
        area=BBOX[2]*BBOX[3]
        areas.append(area)

    mario_real_bbox=mario_BBoxes[np.argmax(areas)]
    return(mario_real_bbox)

def Build_Negative_Lines(positives, image_name):
    Negative_Lines=[]

    for rect_boxes in positives:
        if not rect_boxes:
            Negative_Lines.append(image_name+'\n')
        else:
            Negative_Lines.append('')
          
    return(Negative_Lines)

def Build_Positive_Lines(positives, image_name):
    Positive_Lines=[]

    for BBOX_List in positives:
        if not BBOX_List: 
            Positive_Lines.append('')
            continue

        line=f'{image_name}  {len(BBOX_List)}'

        if not isinstance(BBOX_List, list):
            BBOX_List=[BBOX_List]

        for BBox in BBOX_List:
            line+=f'  {BBox[0]} {BBox[1]} {BBox[2]} {BBox[3]}'
        
        line+='\n'

        Positive_Lines.append(line)
    
    return(Positive_Lines)

def Calculate_Positive_GDs(segmented_image, image_name):
    positives=[]

    for class_number in range(len(colors)):

        binary_image=np.where(segmented_image==class_number, 255, 0).astype(np.uint8)

        c,h=cv.findContours(binary_image, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
        
        positives_for_class=[]

        for contour in c:
            bounding_rectangle=cv.boundingRect(contour)

            height=bounding_rectangle[2]
            width=bounding_rectangle[3]

            if(width * height < ACCEPTABLE_DETECTION_LOWER_THRESHOLD) or (height * width > ACCEPTABLE_DETECTION_HIGHER_THRESHOLD): continue

            if(height/width<ACCEPTABLE_HEIGHT_WIDTH_RATION) or (width/height<ACCEPTABLE_HEIGHT_WIDTH_RATION): continue

            positives_for_class.append(bounding_rectangle)
        
        positives.append(positives_for_class)

    return(positives)

def Criteria(item):
    num_string=sub('[^0-9]', '', item)
    return(int(num_string))

def Initialize_Files(Classes_Names):
    positive_file_names=[]
    negative_file_names=[]
    positive_files=[]
    negative_files=[]

    for class_name in Classes_Names:
        positive_file_names.append(f'MarioNet64 Training\\{class_name}.dat')
        negative_file_names.append(f'MarioNet64 Training\\{class_name}.txt')
        positive_files.append('')
        negative_files.append('')

    return(positive_file_names, negative_file_names, positive_files, negative_files)

def Update_Files(Positive_Files, Negative_Files, Positive_Lines, Negative_Lines):
    
    for i in range(len(Training_Format_Files)):
        Positive_Files[i]+=Positive_Lines[i]
        Negative_Files[i]+=Negative_Lines[i]

    return(Positive_Files, Negative_Files)

def Write_Files(Positive_Files, Negative_Files, Positive_File_Names, Negative_File_Names):
    for i in range(len(Training_Format_Files)):
        with open(Positive_File_Names[i], 'w') as Positive_File:
            Positive_File.write(Positive_Files[i]) 

        with open(Negative_File_Names[i], 'w') as Negative_File:
            Negative_File.write(Negative_Files[i]) 

def Create_Dataset(segmented_folder_name, unsegmented_folder_name):
    segmented_images = sorted(listdir(abspath(segmented_folder_name)), key=Criteria)
    
    unsegmented_images = sorted(listdir(abspath(unsegmented_folder_name)), key=Criteria)

    images=list(zip(segmented_images, unsegmented_images))

    Positive_File_Names, Negative_File_Names, Positive_Files, Negative_Files = Initialize_Files(Training_Format_Files)

    for segmented_image_name, unsegmented_image_name in images:
        segmented_image_name=segmented_folder_name+'\\'+segmented_image_name
        color_image_name=unsegmented_folder_name+'\\'+unsegmented_image_name
        
        segmented_image = cv.imread(segmented_image_name, cv.COLOR_BGR2GRAY)

        Positives = Calculate_Positive_GDs(segmented_image, color_image_name)
        Positives[0] = filter_marios(Positives)
        
        Negative_Lines = Build_Negative_Lines(Positives, color_image_name)
        Positive_Lines = Build_Positive_Lines(Positives, color_image_name)

        Positive_Files, Negative_Files=Update_Files(Positive_Files, Negative_Files, Positive_Lines, Negative_Lines)

    Write_Files(Positive_Files, Negative_Files, Positive_File_Names, Negative_File_Names)
        
if __name__=='__main__':
    Create_Dataset('MarioNet64 segmentation','MarioNet64 Color')


