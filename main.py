
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:29:58 2024

@author: Francois El-Daher
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit
import reconPhaseTieTvWeightedBatch
import genFourierWeights
import tieOtf
from skimage import img_as_float
import cv2

# Initialize lists to store the minimum standard deviations and in-focus filenames
min_std = []
infoc_filename = []

# Get directory to folder with images
#path = "/Users/fwaharte/Documents/Analysis/Density mapping/QPI_Matlab/Data/SHE4_20072023_water"
#path ="/Users/feldaher/Documents/SwainLab/Density mapping/data_yeast/SHE4_20072023_water"
path = "/Users/fwaharte/Documents/Analysis/Density_mapping/QPI_Matlab/Data/Msn2_220923/"
path="/home/jupyter-francois/data/density_smallset"


if not os.path.exists(path):
    print("Path not found")
else:
    print(f'User selected {path}')

#os.chdir(path)
files = [f for f in os.listdir(path) if (f.endswith('.tif') or f.endswith('.tiff'))]
files.sort()

print("first file: ", files[0])

z_stack = 5  # Number of images per stack
number_timepoints = len(files) // z_stack  # Number of timepoints in folder


# TIE QPI
z_step = 1  # Corresponding to 500 nm between images in z-stack

# Physical parameters

data= {'nu': 512e-9, 'mag': 60, 'px': 108.3e-9, 'NA': 1.2}
# wavelength in meters, magnification, pixel size in meters, numerical aperture

# wave number (by definiton)
data['k'] = 2 * np.pi / data['nu']
# maximal frequency
data['fmax'] = 1 / (2 * max(data['px'] / data['mag'], data['nu'] / (2 * data['NA'])))
# distance between images 500 nm
# distance to most out of focus image 1500 nm
data['dzNear']= 0.5e-6
data['dzFar']= 1.5e-6
# correction value for the singularity at the origin
data['sval']= 1
# number of iterations
data['T']= 250
# relative-error tolerance
data['tol'] = 1e-12

# Reconstruction settings
LAMBDA = [2.5e-3]
minRadian = -10
maxRadian = 30
bitsize = 16
maxIntensity = (2 ** bitsize) - 1



# Initialize list to store image filenames
images = np.empty((z_stack, number_timepoints, 1200, 1200), dtype=np.uint16)


os.makedirs('results_python', exist_ok=True)
resdir = os.path.join(path, 'results_python')

for n in range(number_timepoints):
    print('-------------------------------------------------------')
    print('Phase Imaging via Transport-of-Intensity Approach')
    print('-------------------------------------------------------')
    print('Variations of intracellular density during the cell cycle arise from tip-growth regulation in fission yeast')
    print('-------------------------------------------------------\n')

    for q in range(z_stack):
        i = n * z_stack + q
        # Read the image file
        images[q, n,:,:] = plt.imread(os.path.join(path, files[i]))
        print(n, i)


    # Load images for analysis

    Im4=images[0, n,:,:]
    Im5=images[1,n,:,:]
    I0=images[2,n,:,:]
    Im7=images[ 3,n,:,:]
    Im8=images[4,n,:,:]

    print('::: Data is loaded!')


    Im8 = img_as_float(Im8)
    Im7 = img_as_float(Im7)
    Im4 = img_as_float(Im4)
    Im5 = img_as_float(Im5)
    I0 = img_as_float(I0)

    y2 = (Im8 - Im4) / I0
    y1 = (Im7 - Im5) / I0

    # observations indicate which axial derivative is closest and most distant
    data['bNear'] = y1
    data['bFar'] = y2

    # data dimensions
    data['dims'] = data['bNear'].shape

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%Reconstruction
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #construct filters
    [data['W_low'], data['W_high']] = genFourierWeights.genFourierWeights(data['dims'][1],data['dims'][0],data['dzNear'],data['mag'], data['nu'], data['fmax'], 5e-1, 3.1416 / 10, True)
    #forward operators (as Fourier multipliers)
    data['tieOtfNear'] = tieOtf.tie_otf(data['dims'][1],data['dims'][0],data['fmax'],data['nu'],data['dzNear'])

    data['tieOtfFar'] = tieOtf.tie_otf(data['dims'][1],data['dims'][0],data['fmax'],data['nu'],data['dzFar'])

    VERBOSE = True
    ###############################
    #  WTIE-TV Reconstruction
    ###############################
    PHI_TIE = reconPhaseTieTvWeightedBatch.recon_phase_tie_tv_weighted_batch(data, LAMBDA, VERBOSE);

    for ii in range(len(LAMBDA)):
        lambda_val = LAMBDA[ii]

        phi =  np.array(PHI_TIE[0][ii])

        # Adjust dynamic range for saved images
        FinalImage = np.uint16((phi + abs(minRadian)) * maxIntensity / (maxRadian - minRadian))

        cv2.imwrite(os.path.join(resdir, f'Tie_{n:03d}_{minRadian}_{maxRadian}.tif'), FinalImage)



# Background correction?




