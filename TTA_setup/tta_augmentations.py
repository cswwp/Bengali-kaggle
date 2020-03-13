import os
import cv2
import numpy as np
import pandas as pd
import albumentations
from PIL import Image, ImageOps, ImageEnhance
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations import functional as F
from functools import partial 


def rotate(magnitude, image):

    height, width = image.shape[:2]
    cx, cy = width // 2, height // 2

    transform = cv2.getRotationMatrix2D((cx, cy), -magnitude, 1.0)
    dst = cv2.warpAffine( image.copy(), transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return dst


def scale( magnitude , image):
    s = magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [s,0,0],
        [0,s,0],
    ],np.float32)
    dst = cv2.warpAffine( image.copy(), transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return dst

def shear_x( image, magnitude=0.5 ):
    sx = magnitude / 2
    height, width = image.shape[:2]
    transform = np.array([
        [1,sx,0],
        [0,1,0],
    ],np.float32)
    dst = cv2.warpAffine( image.copy(), transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return dst

def shear_y( image, magnitude=0.5 ):
    sy = magnitude / 2

    height, width = image.shape[:2]
    transform = np.array([
        [1, 0,0],
        [sy,1,0],
    ],np.float32)
    dst = cv2.warpAffine( image.copy(), transform, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return dst

# def contrast(image, magnitude=0.5):
#     alpha = magnitude
#     image = image.astype(np.float32) * alpha
#     image = np.clip(image,0,1)
#     return image

def brightness(image , magnitude = 0.5):
    dst = cv2.add(image.copy() , magnitude)
    return dst 

def contrast(image , magnitude = 0.5):
    dst = cv2.multiply(image.copy()  , magnitude)
    return dst 

def blur(image , magnitude = 0.5):
    dst = cv2.GaussianBlur(image.copy() , (magnitude,magnitude ), 0)
    return dst 

def Sharpness(image , magnitude = 0.5):
    blur_im = cv2.GaussianBlur(image.copy() , (magnitude,magnitude ), 0)
    im = cv2.addWeighted(image.copy(), 2, blur_im, -1, 0)
    return im 

def erode(image, magnitude=0.5):
    s = magnitude
    # s = int(round(1 + np.random.uniform(0,1)*magnitude*6))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s,s)))
    dst  = cv2.erode(image.copy(), kernel, iterations=1)
    return dst

def dilate(image, magnitude=0.5):
    s =magnitude
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s,s)))
    dst  = cv2.dilate(image.copy() , kernel, iterations=1)
    return dst

def translate(image , tx= 0, ty=0):
    rows,cols = image.shape
    M = np.float32([[1,0,tx],[0,1,ty]])
    dst = cv2.warpAffine(image.copy(),M,(cols,rows))
    return dst

def clip_blur(image, type= 1, magnitude = 0.5):

    dst = image.copy()

    if type ==1:
        dst[:64,:] =cv2.multiply(image[:64,:].copy() , magnitude)
    elif type ==-1:
        dst[64:,:] =cv2.multiply(image[64:,:].copy() , magnitude)
    
    elif type ==2:
        dst[:,:64] =cv2.multiply(image[:,:64].copy() , magnitude)
    elif type ==-2:
        dst[:,64:] =cv2.multiply(image[:,64:].copy() , magnitude)

    elif type ==3:
        dst[:64,:64] =cv2.multiply(image[:64,:64].copy() , magnitude)
    elif type ==-3:
        dst[64:,:64] =cv2.multiply(image[64:,:64].copy() , magnitude)

    elif type ==4:
        dst[:64,64:] =cv2.multiply(image[:64,64:].copy() , magnitude)
    elif type ==-4:
        dst[64:,64:] =cv2.multiply(image[64:,64:].copy() , magnitude)
        
    return dst


augment=    [   # Rotations
                partial(rotate,magnitude=15),
                partial(rotate,magnitude=-15),
                partial(rotate,magnitude=30),
                partial(rotate,magnitude=-30),
                partial(rotate,magnitude=45),
                partial(rotate,magnitude=-45),

                # Scaling
                partial(scale,magnitude=1.02),
                partial(scale,magnitude=1.04),
                partial(scale,magnitude=1.06),
                partial(scale,magnitude=1.08),
                partial(scale,magnitude=1.1),
                partial(scale,magnitude=1.2),
                partial(scale,magnitude=0.95),
                partial(scale,magnitude=0.9),
                partial(scale,magnitude=0.85),
                partial(scale,magnitude=0.8),
                partial(scale,magnitude=0.7),
                partial(scale,magnitude=0.6),
                partial(scale,magnitude=0.5),

                # Shear
                partial(shear_x,magnitude=0.1),
                partial(shear_x,magnitude=0.2),
                partial(shear_x,magnitude=0.3),
                partial(shear_x,magnitude=0.4),
                partial(shear_x,magnitude=0.5),
                partial(shear_x,magnitude=0.6),

                partial(shear_y,magnitude=0.1),
                partial(shear_y,magnitude=0.2),
                partial(shear_y,magnitude=0.3),
                partial(shear_y,magnitude=0.4),
                partial(shear_y,magnitude=0.5),
                partial(shear_y,magnitude=0.6),

                partial(contrast,magnitude=1.1),
                partial(contrast,magnitude=1.2),
                partial(contrast,magnitude=1.3),
                partial(contrast,magnitude=1.4),
                partial(contrast,magnitude=1.5),
                partial(contrast,magnitude=0.9),
                partial(contrast,magnitude=0.8),
                partial(contrast,magnitude=0.7),
                partial(contrast,magnitude=0.6),
                partial(contrast,magnitude=0.5),


                # Morphology
                partial(erode,magnitude =3),
                partial(erode,magnitude =5),
                # partial(erode,magnitude =7),
                # partial(erode,magnitude =9),

                partial(dilate,magnitude =3),
                partial(dilate,magnitude =5),
                partial(dilate,magnitude =7),
                partial(dilate,magnitude =9),

                # Brightness
                partial(brightness,magnitude =30),
                partial(brightness,magnitude =20),
                partial(brightness,magnitude =10),
                partial(brightness,magnitude =5),
                partial(brightness,magnitude =-30),
                partial(brightness,magnitude =-20),
                partial(brightness,magnitude =-10),
                partial(brightness,magnitude =-5),

                # Blur
                partial(blur,magnitude =3),
                partial(blur,magnitude =5),
                partial(blur,magnitude =7),
                partial(blur,magnitude =9),

                # Sharpness
                partial(Sharpness,magnitude =3),
                partial(Sharpness,magnitude =5),
                partial(Sharpness,magnitude =7),
                partial(Sharpness,magnitude =9),
                partial(Sharpness,magnitude =11),
                partial(Sharpness,magnitude =13),
                partial(Sharpness,magnitude =15),

                # translate
                partial(translate, tx =5),
                partial(translate, tx =10),
                partial(translate, tx =15),
                partial(translate, tx =-10),
                partial(translate, tx =-15),
                partial(translate, tx =-5),
                
                partial(translate, ty =5),
                partial(translate, ty =10),
                partial(translate, ty =15),
                partial(translate, ty =-10),
                partial(translate, ty =-15),
                partial(translate, ty =-5),
                
                partial(clip_blur , type=1),
                partial(clip_blur , type=2),
                partial(clip_blur , type=3),
                partial(clip_blur , type=4),

                partial(clip_blur , type=-1),
                partial(clip_blur , type=-2),
                partial(clip_blur , type=-3),
                partial(clip_blur , type=-4),
                ]
