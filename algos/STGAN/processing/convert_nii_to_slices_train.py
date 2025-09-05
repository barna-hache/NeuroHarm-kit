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
import matplotlib
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('--load_path', type=str, default='test',
                    help='file path where the npy file from the 3 orientations of harmonization files were located')
parser.add_argument('--save_path', type=str, default='test',
                    help='file path where the outputs are saved')

args = parser.parse_args()
print(args)


data_dir = args.load_path + '/'
save_path = args.save_path + '/image/'
os.makedirs(save_path)
 
thr = 0.1

id = 0

for f in sorted(listdir(data_dir)):
    
    print(f)
    f1 = os.path.splitext(f)
    f2 = os. path. splitext(f1[0])

    img = nib.load(data_dir + f)
    a = img.get_fdata()
    a[np.where(a<0)] = 0
    
    cl = []
    for s in range(1, a.shape[2]-1):
        
        b = a[:, :, s]/a.max()
        b = b[..., np.newaxis]
        bn = a[:, :, s-1]/a.max()
        bn = bn[..., np.newaxis]
        bp = a[:, :, s+1]/a.max()
        bp = bp[..., np.newaxis]
        
        if b.sum() > thr and bp.sum() > thr and bn.sum() > thr:

            c = np.concatenate((bn,b,bp), axis=2)
            savename = save_path + str(id).zfill(9) + '.png'#str(id).zfill(6) + '.png' 
            matplotlib.image.imsave(savename, c)
        id += 1
        
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
        
        if b.sum() > thr and bp.sum() > thr and bn.sum() > thr:

            c = np.concatenate((bn,b,bp), axis=2)
            savename = save_path + str(id).zfill(9) + '.png'#str(id).zfill(6) + '.png' 
            matplotlib.image.imsave(savename, c)
        id += 1
       
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
        
        if b.sum() > thr and bp.sum() > thr and bn.sum() > thr:

            c = np.concatenate((bn,b,bp), axis=2)
            savename = save_path + str(id).zfill(9) + '.png'#str(id).zfill(6) + '.png' 
            matplotlib.image.imsave(savename, c)
        id += 1

    
    
