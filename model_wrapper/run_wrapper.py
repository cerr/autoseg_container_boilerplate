# Standard system imports
import os
import json
import glob
import csv
import logging
import time
##

import h5py

import numpy as np

from scipy import io

# Standard routines for loading CERR H5
import CERRPyCo as CERR

#import DL dependencies

# uncomment to load pytorch
#import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#import torch.nn.functional as F

# end module import statements  

########## set filename of model weights
modelWeightsFile = ''

## input/output directories
dataPath = '/scratch'
inputH5Path = dataPath + '/inputH5'
outputH5Path = dataPath + '/outputH5'

softwarePath = '/software'
modelWeightsPath = os.path.join(softwarePath,'model',modelWeightsFile)


## set algorithm_name for use in logging
algorithm_name = ''

## configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(algorithm_name) 


## probability threshold
labelthresh = 0.0

## image slice dimension, pre-defined in alogrithm json

## Load network weights, get this from config
logger.info('Loading saved weights from ' + modelWeightsPath + '...')
model = ''

try:
    #model.load_state_dict(torch.load(modelWeightsPath))
except:
    logger.error(modelWeightsPath + 'not found, unable to load weights')

logger.info('Finish loading weight')



num_labels = len(config['strNameToLabelMap'])
logger.debug('Expected number of structure labels: ' + str(num_labels))

#load the data; this function will load single 3D h5 or multi 2D h5 files
scan_vol, h5file0 = CERR.loadCERRH5(inputH5Path, config)

vol_shape = scan_vol.shape
num_slices = vol_shape[0]
input_size = vol_shape[1:]
logger.debug('Opened data shape: ' + str(vol_shape))
logger.info('Number of slices in loaded data: ' + str(num_slices))

### min-max normalize
logger.info('Applying min-max normalization (-1,1)')

norm_scan=normalize_data(np.rot90(np.flip(scan_vol,axis=1),k=3,axes=(1,2)))

logger.debug('norm_scan shape: ' + str(norm_scan.shape))

### Initialize array to hold model output in same shape as input original_scan_vol
labels_out = np.zeros((num_slices,input_size[0],input_size[1]))

### Begin processing
logger.info('Starting inference...')

#if python:
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for i in range(0,num_slices):

        logger.info('Slice number ' +  str(i + 1) + ' of ' + str(num_slices))

        # orient slice
        inputslice = np.ascontiguousarray(norm_scan[i].reshape(1,1,input_size[0],input_size[1]))
        print(inputslice.shape)

        # Convert numpy array to tensor, which used in pytorch (Jue)
        inputslice=torch.from_numpy(inputslice).cuda().float()
       
        logger.debug('Single slice oriented, rotated 270 degrees, final shape: ' + str(inputslice.shape))

        label_array_probability = model(inputslice)
        label_array_probability = F.softmax(label_array_probability[0],dim=1)
        # convert tensor into numpy array
        label_array_probability = label_array_probability[0].cpu().numpy()
        logger.debug('Size of returned label array struct: ' + str(label_array_probability.shape))

        logger.debug('Thresholding label probability array by ' + str(labelthresh))
        labelslice = np.argmax(label_array_probability, axis = 0)
        logger.debug('size of resulting labelslice ' + str(labelslice.shape))
        labels_out[i] = np.rot90(labelslice.reshape(1,input_size[0],input_size[1]), k = 1, axes = (1,2))

### Save output
### permute axes =-O
### set output data filename

maskOut = np.moveaxis(np.flip(labels_out,axis=1),[0,2],[2,0])
maskH5out = h5file0.replace('scan', 'MASK').replace('input','output')

try:
    with h5py.File(maskH5out, 'w') as h5out:
        h5out.create_dataset("mask", data=maskOut)
    logger.info('Output label H5 file saved to: ' + maskH5out)
except:
    logger.error('Unable to save H5 file output.')

