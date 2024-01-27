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


class_ids =['Mario', 'Goomba', 'Chain Chomp', 'Bobomb', 'Bobomb King', 'Twhomp', 'Whomp', 'Piranha']
class_mapping = dict(zip(range(len(class_ids)), class_ids))

def load_img(filename, target_size):
    img = tf.keras.utils.load_img(filename, target_size=target_size) 
    img = tf.keras.utils.img_to_array(img) 

    return (img)

def Get_BBOX(row):
    xmin=int(row['xmin'])
    ymin=int(row['ymin'])
    xmax=int(row['xmax'])
    ymax=int(row['ymax'])

    bbox=np.array([xmin, ymin, xmax, ymax])

    return bbox

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):    
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )

def Progres_bar(i, num):
    sys.stdout.write('\r' + ' ' * 70 + '\r')
    sys.stdout.flush()
    progress=(i/num)
    Bar_Progress=floor(progress*30)
    Remaining_Bar=30-Bar_Progress
    print('['+Bar_Progress*'='+' '*Remaining_Bar+']'+str(100*progress)+'% concluido',end='', flush=True)

def Unpack_Raw_Format(Images, Boxes, bb_format):
    image = Images
  
    boxes = keras_cv.bounding_box.convert_format(Boxes['boxes'], images=image, source='xywh', target=bb_format)
    print(Boxes['classes'])
    bounding_boxes = {
        "classes": Boxes['classes'],
        "boxes": boxes,
    }
     
    return {"images": image, "bounding_boxes":bounding_boxes} 

def Data_Loader(annotation_file):
    data=pd.read_csv(annotation_file)
    data=data.drop('Unnamed: 0', axis=1, errors='ignore')

    data_groups=data.groupby('filename')
    Dataset={'images':[], 'bounding_boxes':[]}
  
    ngroups=len(data_groups)
    i=0
    for image_name, groups in data_groups:
        image=load_img(image_name, target_size=(224, 224))

        rows=[]
        for _, row in groups.iterrows():
            rows.append(row['bounding_boxes'])

        boxes_coords=ast.literal_eval(rows[0])
        boxes_classes=ast.literal_eval(rows[1])

        BBox_Data_Per_image={'boxes':boxes_coords, 'classes':boxes_classes}
        Dataset['images'].append(tf.data.Dataset.from_tensor_slices(image))
        Dataset['bounding_boxes'].append(tf.data.Dataset.from_tensor_slices(BBox_Data_Per_image))

        i+=1
        Progres_bar(i,ngroups)

        if(i/ngroups>0.005):break

    Built_DBB=Dataset['bounding_boxes'][0]
    Built_DBI=Dataset['images'][0]
  
    for i in range(1, len(Dataset['bounding_boxes'])):
        Built_DBB=Built_DBB.concatenate(Dataset['bounding_boxes'][i])
        Built_DBI=Built_DBI.concatenate(Dataset['images'][i])
    
    Dataset=tf.data.Dataset.zip((Built_DBI, Built_DBB))
    #Dataset=tf.data.Dataset.from_tensor_slices((Dataset['images'], Dataset['bounding_boxes']))
    print('\n')
    return(Dataset)

if __name__=='__main__':
    Dataset=Data_Loader('../Bounding Box Annotations.csv')
    print(Dataset)
    Dataset=Dataset.map(lambda x, y: Unpack_Raw_Format(x,y ,'xyxy'))
    print(Dataset)
    Dataset=Dataset.ragged_batch(5)
    #print(Dataset)
    visualize_dataset(Dataset, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2)
