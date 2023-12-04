#https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth):
    return 1 - dice_coef(y_true, y_pred, smooth)


def dice_coef_multilabel(y_true, y_pred, M=4, smooth=100):
    dice = 0
    for index in range(M):
        dice += dice_coef(y_true[:,:,index], y_pred[:,:,index], smooth)
    return dice 


def multilabel_dice_coef_loss(y_true, y_pred,M=4,smooth=100):
    return 1 - dice_coef_multilabel(y_true, y_pred, smooth)