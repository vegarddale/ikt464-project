
import os
from PIL import Image
import numpy as np

labels_path = "./data/labels/train_v2"
output_path = "./data/images/train_v2"
images_path = "./data/test_2"

os.makedirs(output_path, exist_ok=True)

labels = [f for f in os.listdir(labels_path) if f.endswith(".npy")]

for label in labels:
    input_png_path = os.path.join(images_path, os.path.splitext(label)[0] + ".png")
    output_png_path = os.path.join(output_path, os.path.splitext(label)[0] + ".png")
    img = Image.open(input_png_path)
    img.save(output_png_path)
    
