import tensorflow_datasets as tfds
import tensorflow as tf
import keras
import keras_cv
import numpy as np
from keras_cv import bounding_box, visualization
import os
import tqdm
import sys
from math import floor
import pandas as pd
import ast
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec


class_ids =['Mario', 'Goomba', 'Chain Chomp', 'Bobomb', 'Bobomb King', 'Twhomp', 'Whomp', 'Piranha']
class_mapping = dict(zip(range(len(class_ids)), class_ids))

MAX_CLASSES=len(class_ids)

#Hyperparameters of the network
LEARNING_RATE=0.005
MOMENTUM=0.9
GLOBAL_CLIP=10.0
NUM_EPOCHS=1

#Batch information
SPLIT_RATIO=0.2
BATCH_SIZE=4

#Font size for displaying class name
FONT_SCALE=0.5
#Color for displaying predicted bounding boxes
PREDICTED_COLOR=[0,0,0,0]
#Color for displaying True bounding boxes
TRUE_COLOR=[255,255,255,255]
#Information for augmentation layers
TARGET_SIZE=(224, 224)
SHEAR_FACTOR_X=0.2
SHEAR_FACTOR_Y=0.2
SCALE_FACTOR=(0.75, 1.3)

#Information of the bounding box format(here we have it in X,Y,Width and Height)
BBOX_FORMAT="xywh"

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return tf.cast(image, dtype=tf.float32)    

def Progres_bar(i, num):
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    sys.stdout.flush()
    progress=(i/num)
    Bar_Progress=floor(progress*30)
    Remaining_Bar=30-Bar_Progress
    print('['+Bar_Progress*'='+' '*Remaining_Bar+']'+str(100*progress)+'% concluido',end='', flush=True)

def Unpack_Raw_Format(Image_Path, Boxes, Labels):
    image = load_image(Image_Path)

    bounding_boxes = {
        "classes": tf.cast(Labels,dtype=tf.float32),
        "boxes": Boxes
    }

    return {"images": image, "bounding_boxes": bounding_boxes} 

def Data_Loader(annotation_file):
    data=pd.read_csv(annotation_file)
    data=data.drop('Unnamed: 0', axis=1, errors='ignore')

    data_groups=data.groupby('filename')
    BBoxes=[]
    Labels=[]
    Images=[]
  
    ngroups=len(data_groups)
    i=0
    max_box=0
    for image_name, groups in data_groups:
        rows=[]
        for _, row in groups.iterrows():
            rows.append(row['bounding_boxes'])

        boxes_coords=rows[0]
        boxes_coords=boxes_coords.replace('(', '[')
        boxes_coords=boxes_coords.replace(')', ']')
        boxes_coords=ast.literal_eval(boxes_coords)
      
        boxes_classes=ast.literal_eval(rows[1])

        max_box=max(len(boxes_classes), max_box)

        BBoxes.append(boxes_coords)
        Labels.append(boxes_classes)
        Images.append(image_name)
     
        i+=1
        Progres_bar(i,ngroups)

    BBoxes=tf.ragged.constant(BBoxes)
    Labels=tf.ragged.constant(Labels)
    Images=tf.ragged.constant(Images)
    
    Dataset=tf.data.Dataset.from_tensor_slices((Images, BBoxes, Labels))

    print('Dados de bounding boxes carregados\n')

    num_val=int(i*SPLIT_RATIO)
    
    return(Dataset, num_val, max_box)

augmenter = keras.Sequential(
    layers=
    [
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=BBOX_FORMAT),
        keras_cv.layers.RandomShear(x_factor=SHEAR_FACTOR_X, y_factor=SHEAR_FACTOR_Y, bounding_box_format=BBOX_FORMAT),
        keras_cv.layers.JitteredResize(target_size=TARGET_SIZE, scale_factor=SCALE_FACTOR, bounding_box_format=BBOX_FORMAT),
    ]
)

resizing = keras_cv.layers.JitteredResize(
    target_size=TARGET_SIZE,
    scale_factor=SCALE_FACTOR,
    bounding_box_format=BBOX_FORMAT,
)

def dict_to_tuple(inputs, max_box):
    return inputs["images"], bounding_box.to_dense(inputs["bounding_boxes"], max_boxes=max_box)

def Prep_For_Fitting(Data, max_box):
    Data = Data.map(lambda x: dict_to_tuple(x, max_box), num_parallel_calls=tf.data.AUTOTUNE)
    Data = Data.prefetch(tf.data.AUTOTUNE)

    return(Data)

def Get_Data(Data_Loaded, aug_function):
    Data = Data_Loaded.map(Unpack_Raw_Format, num_parallel_calls=tf.data.AUTOTUNE)
    Data = Data.shuffle(BATCH_SIZE * 4)
    Data = Data.ragged_batch(BATCH_SIZE, drop_remainder=True)
    Data = Data.map(aug_function, num_parallel_calls=tf.data.AUTOTUNE)

    return(Data)


def train(val_data, train_data, maxbox):
    train_data=Prep_For_Fitting(train_data, maxbox)
    val_data=Prep_For_Fitting(val_data, maxbox)
    
    backbone = keras_cv.models.MobileNetV3Backbone.from_preset("mobilenet_v3_small_imagenet",load_weights=False,)
    optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, global_clipnorm=GLOBAL_CLIP)

    coco_metrics_callback = keras_cv.callbacks.PyCOCOCallback(val_data, bounding_box_format=BBOX_FORMAT)
    model=keras_cv.models.YOLOV8Detector(num_classes=MAX_CLASSES, backbone=backbone, bounding_box_format=BBOX_FORMAT,  fpn_depth=1)
    model.compile(optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou")
    model.fit(train_data, epochs=NUM_EPOCHS,callbacks=[coco_metrics_callback])
    model.save('../MarioNet64 Detector.keras')

    return model


def Visualize_BBoxes(BBoxes, Image, Labels, color):
    I=cv.cvtColor(np.uint8(Image.numpy()), cv.COLOR_RGB2BGR)
    
    font = cv.FONT_HERSHEY_SIMPLEX 

    BBoxes=np.int64(BBoxes.numpy())
    Labels=np.int64(Labels.numpy())

    for Box, Label in list(zip(BBoxes, Labels)):
        if(Box[0]>=0 and Box[1]>=0 and Box[2]>=0 and Box[3]>=0 and Label>=0):
            I=cv.rectangle(I, Box, color)
            I=cv.putText(I, class_mapping[Label], (Box[0], Box[1]), font, FONT_SCALE, color)

    return(I)

#Plots images with bounding boxes shown
#Both predicted and ground truth
def Visualize_Predicted(Val_Data, max_box, model, rows, cols, Title):
    Val_Data=Prep_For_Fitting(Val_Data, max_box)
    Images, Boxes=next(iter(Val_Data.take(1)))

    Predicted_Boxes=model.predict(Images)
    Boxes=bounding_box.to_ragged(Boxes)
    Predicted_Boxes=bounding_box.to_ragged(Predicted_Boxes)
    True_Boxes, True_Labels=Boxes['boxes'], Boxes['classes'], 
    Predicted_Boxes, Predicted_Labels=Predicted_Boxes['boxes'], Predicted_Boxes['classes']

    fig = plt.figure(figsize=(12, 12))
    plt.title(Title)
    plt.axis('off')
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols),axes_pad=0.05,label_mode="1")
    
    for Image,True_BBox_list, True_Label_list, Predicted_BBox_List, Predicted_Label_List, ax in list(zip(Images, True_Boxes, True_Labels, Predicted_Boxes, Predicted_Labels, grid)):
        I=Visualize_BBoxes(True_BBox_list, Image, True_Label_list, TRUE_COLOR)
        I=Visualize_BBoxes(Predicted_BBox_List, Image, Predicted_Label_List, PREDICTED_COLOR)
        ax.imshow(I, origin="lower")
        ax.axis('off')
    
    plt.show()


if __name__=='__main__':
    Dataset, num_val, max_box=Data_Loader('../Bounding Box Annotations.csv')

    val_data = Dataset.take(num_val)
    train_data = Dataset.skip(num_val)

    val_data=Get_Data(val_data, resizing)
    train_data=Get_Data(train_data, augmenter)

    model=train(val_data, train_data, max_box)

    Visualize_Predicted(val_data, max_box, model, 2, 2, "Bounding Boxes Preditas x reais")
