import tensorflow_datasets as tfds
import tensorflow as tf
import keras
import keras_cv
import numpy as np
from keras_cv import bounding_box
import os
from keras_cv import visualization
import tqdm
import sys
from math import floor
import pandas as pd


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


def Unpack_Raw_Format(Data, bb_format):
    image = Data['images']

    boxes = keras_cv.bounding_box.convert_format(Data['bounding_boxes']['boxes'], images=image, source='xyxy', target=bb_format)

    bounding_boxes = {
        "classes": Data['bounding_boxes']['classes'],
        "boxes": boxes,
    }
     
    return {"images": image, "bounding_boxes":boxes} 

def Data_Loader(annotation_file):
    data=pd.read_csv(annotation_file)
    data_groups=data.groupby('filename')

    Dataset={'images':[], 'bounding_boxes':[]}

    ngroups=data_groups.ngroups
    i=0

    for image_name, group in data_groups:
        i+=1
        
        BBoxes={'classes': [], 'boxes':[]}
        for _, row in group.iterrows():
            BBoxes['boxes'].append(Get_BBOX(row))
            BBoxes['classes'].append(class_ids.index(row['class']))

        Dataset['bounding_boxes'].append(BBoxes)
        image=load_img(image_name,(224, 224))
        Dataset['images'].append(image)
       
        sys.stdout.write('\r' + ' ' * 70 + '\r')
        sys.stdout.flush()
        progress=(i/ngroups)
        Bar_Progress=floor(progress*30)
        Remaining_Bar=30-Bar_Progress
        print('['+Bar_Progress*'='+' '*Remaining_Bar+']'+str(100*progress)+'% concluido',end='', flush=True)

        if(100*progress>1): break
    
    
    Dataset=tf.data.Dataset.from_tensor_slices(Dataset)

    print('\n')
 
    return(Dataset)

if __name__=='__main__':
    Dataset=Data_Loader('Bounding Box Annotations.csv')
 
    Dataset=Dataset.map(lambda x: Unpack_Raw_Format(x ,'xyxy'))

    Dataset=Dataset.ragged_batch(5)
    print(Dataset)
    visualize_dataset(Dataset, bounding_box_format="xyxy", value_range=(0, 255), rows=2, cols=2)




{"train": [{"input": [[3, 1, 2], [3, 1, 2], [3, 1, 2]], "output": [[4, 5, 6], [4, 5, 6], [4, 5, 6]]}, 
           {"input": [[2, 3, 8], [2, 3, 8], [2, 3, 8]], "output": [[6, 4, 9], [6, 4, 9], [6, 4, 9]]}]}