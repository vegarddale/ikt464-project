
# =============================================================================
# Separate training and validation data for yolo
# =============================================================================

from skimage.io import imread, imsave  # Import imsave for saving images
import os
import math

images_path = "./data/yolo_v8/images/all_images"
all_labels_path = "./data/yolo_v8/labels/all_labels"
train_path = "./data/yolo_v8/images/train_v1"
train_labels_path = "./data/yolo_v8/labels/train_v1"
val_path = "./data/yolo_v8/images/val_v1"
val_labels_path = "./data/yolo_v8/labels/val_v1"

image_ids = [os.path.splitext(file)[0] for file in os.listdir(all_labels_path)]

train_split = 0.8
test_split = 0.2

train_test_split = int(train_split*len(image_ids))
train_ids = image_ids[:train_test_split]
val_ids = image_ids[train_test_split:]



# save the training images
for image_id in train_ids:
    image = imread(os.path.join(images_path, image_id + ".tif"))  
    out_file_path = os.path.join(train_path, image_id + ".tif")
    imsave(out_file_path, image) 

# save the validation images
for image_id in val_ids:
    image = imread(os.path.join(images_path, image_id + ".tif")) 
    out_file_path = os.path.join(val_path, image_id + ".tif")
    imsave(out_file_path, image) 


import shutil
# copy training labels
for image_id in train_ids:
    source_file = f"{all_labels_path}/{image_id}.txt"
    dst_file = f"{train_labels_path}/{image_id}.txt"
    shutil.copy(source_file, dst_file)
    
# copy validation labels
for image_id in val_ids:
    source_file = f"{all_labels_path}/{image_id}.txt"
    dst_file = f"{val_labels_path}/{image_id}.txt"
    shutil.copy(source_file, dst_file)
    
    

    
    
    
    
    
    
    
    
    