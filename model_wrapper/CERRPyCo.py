import os
import logging
import h5py
import numpy as np
import time
import glob
import json
import cv2
from scipy import ndimage


def min_max_normalize(data_arr, norm_min = 0, norm_max = 1, thresh_min = None, thresh_max = None):
    if thresh_min != None or thresh_max != None:
        data_arr = np.clip(data_arr, a_min = thresh_min, a_max = thresh_max)
    return norm_min + (norm_max - norm_min)*(data_arr - data_arr.min())/(data_arr.max() - data_arr.min())



def loadJSONConfig(inputH5Path):
    configglob = glob.glob(os.path.join(inputH5Path,'*_config.json'))
    with open(configglob[0]) as json_file:
        config = json.load(json_file)
    return config    



def loadCERRH5(inputH5Path, config):
    h5glob = glob.glob(os.path.join(inputH5Path,'*.h5'))
    try:
        offset = config['notes']['preprocessing'][0]['imageOffset']
        print(offset)
    except:
        offset = 0
    if len(h5glob) == 1:
        h5file = h5glob[0]
        s = h5py.File(h5file, 'r')
        scan_vol = np.array(s['scan'][:]) + offset
    elif len(h5glob) > 1:
        h5file = h5glob[0]
        s = h5py.File(h5file, 'r')
        input_size = s['scan'][:].shape
        scan_vol = np.zeros((len(h5glob),input_size[0],input_size[1]))
        for i in range(0,len(h5glob)):
            h5file = h5glob[i]
            s = h5py.File(h5file, 'r')
            in_slice = np.array(s['scan'][:]) + offset
            scan_vol[i] = in_slice.reshape(1,slice_dim[0],slice_dim[1])
    return scan_vol, h5glob[0]
    #if rot90 != 0:
    #    scan_arr = np.rot90(scan_arr, axes=(0,2))
    #if flipud != 0:
    #    scan_arr = np.flipud(scan_arr)



def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def session_id(working_path = ''):
    return os.path.join(working_path,time.time())

#################################


def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth), intersection

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#def load_CERR_h5(h5_list, slicedim=None):
#    h0 = h5_list[0]
#    s = h5py.File(hd5_filename, 'r')
#    scan_arr = np.array(s['scan'][:]) + offset
#    if rot90 != 0:
#        scan_arr = np.rot90(scan_arr, axes=(0,2))
#    if flipud != 0:
#       scan_arr = np.flipud(scan_arr)
#    return original_scan

#def dim(scan_arr):
#    height, width, length = np.shape(scan_arr)
#    return height, width, length



def refineInputSlice(inputslice, label_prob_arr, refidx, pad = 10):
    #assume axis orientation 0 = z (1-dim slice), 1x2 = image slice, 3 = label (by refidx)
    input_size = label_prob_arr.shape[1:3]
    firstpassmask = label_prob_arr[:,:,:,refidx].reshape(input_size[0],input_size[1])
    firstpassmask[firstpassmask > 0] = 1
    if np.count_nonzero(firstpassmask) > 0:
        K = ndimage.find_objects(firstpassmask, max_label = 1)
        idx0 = np.amin(np.array([K[0][0].start,K[0][1].start])) - pad
        if idx0 < 0:
            idx0 = 0 
        idxf = np.amax(np.array([K[0][0].stop,K[0][1].stop])) + pad
        if idxf > input_size[0]:
            idxf = input_size[0]
        cropdim = idxf - idx0
        #logger.debug('Crop dimension ' + str(cropdim))
        cropslice = inputslice[:,idx0:idxf,idx0:idxf,:]
        inputReslice = cv2.resize(cropslice.reshape(cropdim,cropdim), dsize = (input_size[0],input_size[1])).reshape(1,input_size[0],input_size[1],1)
        return [idx0,idxf], inputReslice
    else:
        return None, None


def resliceLabelCrop(labelReslice, idxRe, input_size):
    cropdim = idxRe[1] - idxRe[0]
    labelCropslice = cv2.resize(labelReslice, dsize = (cropdim,cropdim), interpolation=cv2.INTER_NEAREST).reshape(1,cropdim,cropdim)
    labelslice = np.zeros((1,input_size[0],input_size[1]))
    labelslice[:,idxRe[0]:idxRe[1],idxRe[0]:idxRe[1]] = labelCropslice
    return labelslice
