import tensorflow as tf
import cv2 as cv
from math import radians, sqrt, sin
from keras_cv import bounding_box
import numpy as np
import pandas as pd
import ast
import keras
import keras_cv
from os import listdir

RATIOS=[0.75, 3.2, 0.85, 2.2, 2.85 ,3.5, 1.5]
NAMES=['Goomba', 'Chain Chomp', 'Bobomb', 'Bobomb King', 'Twhomp', 'Whomp', 'Piranha']
FRAME_NAME='frame_7770'
ANNOTATIONS_FILE='../Bounding Box Annotations.csv'
FONT_SCALE=0.4
DIMS_ORIGINAIS=(720, 405)
DIMS_REDE=(224, 224)
LEARNING_RATE=0.005
MOMENTUM=0.9
GLOBAL_CLIP=10.0

def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array  

def filter_marios(BBoxes, indexes):
    mario_BBoxes=BBoxes

    if len(mario_BBoxes)<=1: return(mario_BBoxes)

    areas=[]
    for index in indexes:
        BBOX=BBoxes[index]
        rec=cv.boundingRect(BBOX)
        area=rec[2]*rec[3]
        areas.append(area)

    mario_real_bbox=indexes[np.argmax(areas)]

    return(mario_real_bbox)

def Post_Process_Boxes(Predicted_Boxes):
    New_Boxes=[]

    for Box in Predicted_Boxes:
        R_BOX=[Box[0]/ DIMS_REDE[0], Box[1]/DIMS_REDE[1], Box[2]/DIMS_REDE[0], Box[3]/DIMS_REDE[1]]
        New_Boxes.append([R_BOX[0]*DIMS_ORIGINAIS[0], R_BOX[1]*DIMS_ORIGINAIS[1], R_BOX[2]*DIMS_ORIGINAIS[0], R_BOX[3]*DIMS_ORIGINAIS[1]])
    
    return(New_Boxes)

#Calcula as distâncias dos inimigos baseados nas bounding boxes
def Calculate_Distance(Mario_BBox, Enemy_BBox, Enemy_Label):
    Enemy_Label-=1 
    MarioHeight=Mario_BBox[3]
    Enemy_BBox_Height=Enemy_BBox[3]
    X1=Mario_BBox[0]
    X2=Enemy_BBox[0]
    Y1=Mario_BBox[1]+Mario_BBox[3]
    Y2=Enemy_BBox[1]+Enemy_BBox[3]

    Enemy_Expected_height=MarioHeight*RATIOS[Enemy_Label]

    H1=min(Enemy_Expected_height, Enemy_BBox_Height)
    H2=max(Enemy_Expected_height, Enemy_BBox_Height)

    A=radians(30.0)
    B=radians(60.0)
    P1=sin(B)*H1/sin(A)

    P2=H2*P1/H1
    P=P2-P1

    Distancia=sqrt((Y2-Y1)**2+(X2-X1)**2+P**2)

    Distancia_Formatada=f'{NAMES[Enemy_Label]} - {P:.2f}'

    return(Distancia_Formatada)

def Calculate_Distances(Mario_BBox, Boxes, Labels):
    Distances=[]

    for BBox, Label in list(zip(Boxes, Labels)):
        Distances.append(Calculate_Distance(Mario_BBox, BBox, Label))


def Visualize_BBoxes(Image, BBoxes,  Labels_And_Dists, color):    
    I=Image
    font = cv.FONT_HERSHEY_SIMPLEX 

    for Box, Label_And_Dist in list(zip(BBoxes, Labels_And_Dists)):
        if(Box[0]>=0 and Box[1]>=0 and Box[2]>=0 and Box[3]>=0):
            I=cv.rectangle(I, Box, color)
            I=cv.putText(I, Label_And_Dist, (Box[0], Box[1]), font, FONT_SCALE, color)

    return(I)

resize=keras_cv.layers.Resizing(height=224, width=224)

def Find_BBox_Per_Image(Image_Name, Model, Filename):
    Image=get_img_array(image_name, DIMS_REDE)
  
    Boxes=Model.predict(Image)
    Boxes=bounding_box.to_ragged(Boxes)

    Predicted_Boxes, Predicted_Labels=Boxes['boxes'], Boxes['classes']
 
    Predicted_Boxes=np.int64(Predicted_Boxes.numpy())
    Predicted_Labels=np.int64(Predicted_Labels.numpy())

    if Predicted_Boxes.shape[1]==0:
        print("Nenhuma bounding box encontrada")
        return

    Predicted_Boxes=Post_Process_Boxes(Predicted_Boxes)

    Mario_Indexes=np.argwhere(Predicted_Labels==0)

    if(len(Mario_Index)>1):
        Mario_Index=filterMario(Predicted_Boxes, Mario_Index)

    Mario_BBox=Predicted_Boxes[Mario_Index]
    

    Predicted_Boxes=np.delete(Predicted_Boxes, Mario_Indexes)
    Predicted_Labels=np.delete(Predicted_Labels, Mario_Indexes)

    description=[]

    for BBox, Label in list(zip(Predicted_Boxes, Predicted_Labels)):
        description.append(Calculate_Distance(Mario_BBox, BBox, Label))

    description.append('Mario')
    Predicted_Boxes.append(Mario_BBox)

    Image=cv.imread(Image_Name)
    I=Visualize_BBoxes(Image, Predicted_Boxes, description, [255, 0, 255]) 

    cv.imwrite(Filename, I)

def Ground_Truth_BBox(imagename, annotations_file, Filename):
    dataset=pd.read_csv(annotations_file)
    dataset=dataset.drop('Unnamed: 0', axis=1, errors='ignore')
    Image=cv.imread(imagename)

    rows=dataset[dataset['filename']==imagename]
    BBs=rows.iloc[0][1]
    BBs=BBs.replace('(', '[')
    BBs=BBs.replace(')', ']')
    BBs=ast.literal_eval(BBs)
    Labels=ast.literal_eval(rows.iloc[1][1])

    Mario_Index=Labels.index(0)
    Mario_BBox=BBs[Mario_Index]

    Labels.pop(Mario_Index)
    BBs.pop(Mario_Index)

    description=[]

    for BBox, Label in list(zip(BBs, Labels)):
        description.append(Calculate_Distance(Mario_BBox, BBox, Label))

    description.append('Mario')
    BBs.append(Mario_BBox)

    I=Visualize_BBoxes(Image, BBs, description, [255, 0, 255]) 

    cv.imwrite(Filename, I)


def LoadModel():
    backbone = keras_cv.models.MobileNetV3Backbone.from_preset("mobilenet_v3_small_imagenet",load_weights=False,)
    optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, global_clipnorm=GLOBAL_CLIP)

    Model=keras_cv.models.YOLOV8Detector(num_classes=8, backbone=backbone, bounding_box_format="xywh",  fpn_depth=1)
    Model.compile(optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou")
    Model.load_weights('MarioObjectDetector.tf')
    return(Model)

def Try_All_Images(path, Model):
    filelist=listdir(path)
    
    for filename in filelist:
        image_name=f'..\\MarioNet64_Color\\{filename}'
        predicted_name=f'..\\Predicted Boxes\\{filename}'
        Find_BBox_Per_Image(image_name, Model, filename)


if __name__=='__main__':
    image_name=f'..\\MarioNet64_Color\\{FRAME_NAME}.jpg'
    ground_truth_name=f'..\\Ground Truth Boxes\\{FRAME_NAME} GT.jpg'

    Ground_Truth_BBox(image_name, ANNOTATIONS_FILE, ground_truth_name)

    predicted_name=f'..\\Predicted Boxes\\{FRAME_NAME}.jpg'
    
    #Model=LoadModel()

    #Try_All_Images('..\\MarioNet64_Color', Model)



