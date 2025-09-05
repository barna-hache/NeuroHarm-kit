#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:43:12 2020

@author: bigting84
"""
import numpy as np
import nibabel as nib
import os
from os import listdir
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
# import cv2
import argparse



def convert_nii_to_slices(input_img, save_path):
    
    (fpath, f) = os.path.split(input_img)
    id = 0

    f1 = os.path.splitext(f)
    f2 = os. path. splitext(f1[0])
    img_path = save_path + f2[0]
    os.makedirs(img_path)
    img = nib.load(input_img)
    a = img.get_fdata()
    a[np.where(a<0)] = 0
    
    save_pathc = img_path + '/sagittal/'
    os.makedirs(save_pathc)
    save_pathc = save_pathc + '/image/'
    os.makedirs(save_pathc)
    for s in range(1, a.shape[2]-1):    
        b = a[:, :, s]/a.max()
        b = b[..., np.newaxis]
        bn = a[:, :, s-1]/a.max()
        bn = bn[..., np.newaxis]
        bp = a[:, :, s+1]/a.max()
        bp = bp[..., np.newaxis]
        c = np.concatenate((bn,b,bp), axis=2)
        savename = save_pathc + str(id).zfill(9) + '.png'#str(id).zfill(6) + '.png' 
        matplotlib.image.imsave(savename, c)
        id += 1


    save_pathc = img_path + '/coronal/'
    os.makedirs(save_pathc)  
    save_pathc = save_pathc + '/image/'
    os.makedirs(save_pathc)
    for s in range(1, a.shape[1]-1):        
        b = a[:, s, :]/a.max()
        b = b.squeeze()
        b = b[..., np.newaxis]
        bn = a[:, s-1, :]/a.max()
        bn = bn.squeeze()
        bn = bn[..., np.newaxis]
        bp = a[:, s+1, :]/a.max()
        bp = bp.squeeze()
        bp = bp[..., np.newaxis]
        c = np.concatenate((bn,b,bp), axis=2)
        savename = save_pathc + str(id).zfill(9) + '.png'#str(id).zfill(6) + '.png' 
        matplotlib.image.imsave(savename, c)
        id += 1
        

    save_pathc = img_path + '/axial/'
    os.makedirs(save_pathc)
    save_pathc = save_pathc + '/image/'
    os.makedirs(save_pathc)
    for s in range(1, a.shape[0]-1):        
        b = a[s, :, :]/a.max()
        b = b.squeeze()
        b = b[..., np.newaxis]
        bn = a[s-1, :, :]/a.max()
        bn = bn.squeeze()
        bn = bn[..., np.newaxis]
        bp = a[s+1, :, :]/a.max()
        bp = bp.squeeze()
        bp = bp[..., np.newaxis]
        c = np.concatenate((bn,b,bp), axis=2)
        savename = save_pathc + str(id).zfill(9) + '.png'#str(id).zfill(6) + '.png' 
        matplotlib.image.imsave(savename, c)
        id += 1


parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('--input_img', type=str, default='test',
                    help='Input image file, with file path')
parser.add_argument('--save_path', type=str, default='test',
                    help='file path where the outputs are saved')
parser.add_argument('--is_ref', type=bool, default=False,
                    help='indicate whether the input image is the reference image')


args = parser.parse_args() 
print(args)   

save_path = args.save_path
save_path = save_path + '/'
if args.is_ref is True:
    save_path = save_path + 'ref/'
    os.makedirs(save_path)


print('input_image:')
print(args.input_img)

convert_nii_to_slices(args.input_img, save_path)
 
