# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:30:13 2023

@author: vegard
"""

from keras.utils import Sequence
import numpy as np
import os
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
import cv2 as cv

class CustomDataGenerator(Sequence):
    def __init__(self, images_path, labels_path, image_ids, batch_size, image_size, n_classes, logits=False, shuffle=True, transform=False):
        self.images_path = images_path
        self.labels_path = labels_path
        self.image_ids =  image_ids
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.logits=logits
        self.shuffle = shuffle
        self.transform = transform
        self.indexes = np.arange(len(self.image_ids))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_image_ids = self.image_ids[start:end]
        # Load and preprocess the batch of images and labels
        x, y = self.__data_generation__(batch_image_ids)
        if not self.logits:
            for i in range(len(y)):
                y[i] = tf.keras.utils.to_categorical(y[i], self.n_classes)
        if self.transform:
            x, y = self.__transform__(x, y)
        x = np.stack(x)
        y = np.stack(y)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __transform__(self, x, y):
        # Histogram equalization
        # Pixel normalization
        
        # x_t = []
        # for x_i in x:
        #     x_i_equalized = exposure.equalize_hist(x_i)
        #     # x_i_normalized = x_i_equalized.astype(np.float32) / 255.0
        #     x_t.append(x_i_equalized)

        x_t = []
        for x_i in x:
            #Scharr
            gray_img = cv.cvtColor(x_i, cv.COLOR_RGB2GRAY)
            scharr_x = cv.Scharr(gray_img, cv.CV_64F, 1, 0)
            scharr_y = cv.Scharr(gray_img, cv.CV_64F, 0, 1)
            scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
            normalized_scharr = cv.normalize(scharr_magnitude, None, 0, 255, cv.NORM_MINMAX)
            scharr_result = np.uint8(normalized_scharr)
            scharr_img = np.dstack([x_i, scharr_result])
            x_t.append(scharr_img)
            # #laplacian
            # gray_img = cv.cvtColor(x_i, cv.COLOR_RGB2GRAY)
            # laplacian_result = cv.Laplacian(gray_img, cv.CV_64F)
            # normalized_laplacian = cv.normalize(laplacian_result, None, 0, 255, cv.NORM_MINMAX)
            # normalized_laplacian = np.uint8(normalized_laplacian)
            # laplacian_img = np.dstack([x_i, normalized_laplacian])
            # x_t.append(laplacian_img)
    
        return np.array(x_t), y
        
    
    def __data_generation__(self, batch_image_ids):
        X = []
        y = []
        for image_id in batch_image_ids:
            try:
                img = imread(self.images_path + "resized_800_" + image_id + ".png")
                if len(img.shape) == 2:
                    continue
                X.append(img)
                mask = np.load(self.labels_path + image_id + ".npy")
                y.append(mask)
            except Exception as e:
                print(f"Error loading {image_id}: {e}")
        
        return X, y
