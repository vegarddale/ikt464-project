

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "./data/images/train_v2 – Kopi/"
label_path = "./data/labels/train_v2 – Kopi\\"
imgs = [file for file in os.listdir(label_path)]
all_imgs = [file for file in os.listdir(img_path)]

count = 0
for img in imgs:
    image = Image.open(os.path.join(img_path, img.rstrip(".npy") + ".png"))
    gray_image = image.convert("L")
    image_array = np.array(gray_image)
    black_threshold = 15 

    black_pixels = np.sum(image_array < black_threshold)
    
    total_pixels = image_array.size
    black_percentage = (black_pixels / total_pixels) * 100
    

    large_black_threshold = 20
    
    if black_percentage >= large_black_threshold:
        print("The image contains a large portion of black.")
        count += 1
        os.remove(os.path.join(img_path, img.rstrip(".npy") + ".png"))
        os.remove(os.path.join(label_path, img))
    else:
        print("The image does not contain a large portion of black.")
        plt.imshow(image_array)
        plt.axis('off')
        plt.show()

print("deleted_imgs: ", count)
print("nr of imgs: ", len(imgs))
print(len(imgs) - len([file for file in os.listdir(label_path)]))
print(len(all_imgs) - len([file for file in os.listdir(img_path)]))