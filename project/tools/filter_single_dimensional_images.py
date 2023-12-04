
# =============================================================================
# due to some images being completely black they only have 1 dimension after resize
# remove images where there is a single dimension
# =============================================================================

import os
from skimage.io import imread


img_path = "./data/test_2/resized_800"

all_imgs = [file for file in os.listdir(img_path) if file.endswith(".png")]
img_shape = (300, 800, 3)
del_imgs = []
for img in all_imgs:
    a = imread(os.path.join(img_path + " – Kopi (2)", img))
    if a.shape != img_shape:
        del_imgs.append(img)
        os.remove(os.path.join(img_path + " – Kopi (2)", img))
        

all_imgs_2 = [file for file in os.listdir(img_path + " – Kopi") if file.endswith(".png")]
print(len(all_imgs) - len(all_imgs_2))

labels_path = "./data/labels/train_v1_v4_1D"
all_labels =  [file for file in os.listdir(labels_path) if file.endswith(".npy")]
del_imgs = [del_img[12:] for del_img in del_imgs]

count = 0
for del_img in del_imgs:
    del_img = del_img.rstrip(".png") + ".npy"
    if del_img in all_labels:
        count += 1
        os.remove(os.path.join(labels_path + " – Kopi", del_img))
        
    
all_labels_2 =  [file for file in os.listdir(labels_path+ " – Kopi") if file.endswith(".npy")]
print(len(all_labels) - len(all_labels_2))
