import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np

import os
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import gc
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
# from tensorflow.keras_tqdm import TQDMNotebookCallback
from tqdm import tqdm

from matplotlib.ticker import NullFormatter
from utils.map_handler import to_3dmap, to_3dmap_v2

from copy import deepcopy
import cv2
import numpy as np
# from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, ZeroPadding2D
from tensorflow.keras.models import Model
from utils.data_generator import AugmentingDataGenerator
from sklearn.utils import shuffle


def make_colormap(seq):

    ''' *************************************   
    Return a Linear Segmented Colormap. we do that
    in order to plot the images with popular Yellow, blue
    , red colormap which does not exist in python.

    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    ***************************************'''   
    
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
    
def colorit(image, cmap='seismic'):
    
    ''' *************************************   
        This function is designed to
        colorize the arbitrary gray images based
        on a color map that we specify in cmap
        arguement of the function.
    ***************************************:'''   
    cm = plt.get_cmap(cmap)
    colored_image = cm(image/350)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    return colored_image

c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
    [c('blue'), c('yellow'), 0.33, c('yellow'), c('red'), 0.66, c('red'), c('white')])


def load_dataset_mo(data_path, local_path, batch_size = 4):
    masks_all = np.load(data_path + 'train_mask_data_22953.npy')
    maps_all = np.load(data_path + 'train_rnflt_data_22953.npy')

    kernel = np.ones((4,4), np.uint8)
    train_maps = []
    import random
    from PIL import Image 

    for i in range(len(maps_all)):
        img_t_ = cv2.resize(maps_all[i], (256, 256)) 
        img_t = cv2.morphologyEx(img_t_, cv2.MORPH_CLOSE, kernel)
        img = colorit(img_t[:, :], cmap=rvb)
        img = img[6:250, 6:250, :]
        img = cv2.resize(img, (256, 256))
        train_maps.append(img)
                  
    random.shuffle(train_maps)
    train_maps = np.array(train_maps)
        
    val_maps = train_maps[:1000]
    train_maps = train_maps[-20000:]

    print("train size:{}, val size:{}".format(len(train_maps), len(val_maps)))

    # Create training generator
    train_datagen = AugmentingDataGenerator(mask_maps = masks_all, rescale=1./255)
    train_generator = train_datagen.flow(
        train_maps, 
        batch_size=batch_size,
        seed=42)
    
    # Create validation generator
    val_datagen = AugmentingDataGenerator(mask_maps = masks_all, rescale=1./255)
    val_generator = val_datagen.flow(
        val_maps, 
        batch_size=batch_size, 
        seed=42)
    
    
    return train_generator, val_generator, len(train_maps), len(val_maps)
    
    



def load_dataset(data_path, local_path, batch_size = 4):
    
    train_maps_file = os.path.join(local_path, 'train_maps.npy')
    #masks_all = np.load(data_path + 'train_mask_data_22953.npy')
    masks_all = np.load(local_path + 'train_mask_data_22987.npy')
    
    if os.path.exists(train_maps_file):
        train_maps = np.load(train_maps_file)       
        val_maps = train_maps[:1000]
        train_maps = train_maps[-20000:]
        
        print("train size:{}, val size:{}".format(len(train_maps), len(val_maps)))
        
    else:            
        kernel = np.ones((4,4), np.uint8)
        train_maps = []
        
        maps_all = np.load(local_path + 'train_rnflt_data_22953.npy')
        
        import random
        from PIL import Image 

        for i in range(len(maps_all)):  
            img_t = cv2.morphologyEx(maps_all[i], cv2.MORPH_CLOSE, kernel)
            img_t = cv2.resize(img_t, (256, 256))
            img = to_3dmap_v2(img_t)
            img = img[6:250, 6:250, :]
            img = cv2.resize(img, (256, 256))
            train_maps.append(img)
                  
        random.shuffle(train_maps)
        train_maps = np.array(train_maps) 
        
        np.save(train_maps_file, train_maps)
        
        val_maps = train_maps[:1000]
        train_maps = train_maps[-20000:]

        print("train size:{}, val size:{}".format(len(train_maps), len(val_maps)))

    # Create training generator
    train_datagen = AugmentingDataGenerator(mask_maps = masks_all, rescale=1./255)
    train_generator = train_datagen.flow(
        train_maps, 
        batch_size=batch_size,
        seed=42)
    
    # Create validation generator
    val_datagen = AugmentingDataGenerator(mask_maps = masks_all, rescale=1./255)
    val_generator = val_datagen.flow(
        val_maps, 
        batch_size=batch_size, 
        seed=42)
    
    
    return train_generator, val_generator, len(train_maps), len(val_maps)



def load_dataset_tuneparams(data_path, local_path, batch_size = 4):
    
    train_maps_file = os.path.join(local_path, 'train_maps_tuneparams.npy')
    masks_all = np.load(data_path + 'paramtune_mask_data_2k.npy')
    
    if os.path.exists(train_maps_file):
        train_maps = np.load(train_maps_file)       
        val_maps = train_maps[:200]
        train_maps = train_maps[200:]
        
        print("train size:{}, val size:{}".format(len(train_maps), len(val_maps)))
        
    else:            
        kernel = np.ones((4,4), np.uint8)
        train_maps = []
        
        maps_all = np.load(data_path + 'paramtune_rnflt_data_2k.npy')
        
        import random
        from PIL import Image 

        for i in range(len(maps_all)):
            img_t_ = cv2.resize(maps_all[i], (256, 256))
            img_t = cv2.morphologyEx(img_t_, cv2.MORPH_CLOSE, kernel)
            img = to_3dmap(img_t)
            img = img[6:250, 6:250, :]
            img = cv2.resize(img, (256, 256))
            train_maps.append(img)
                  
        random.shuffle(train_maps)
        train_maps = np.array(train_maps) 
        
        np.save(train_maps_file, train_maps)
        
        val_maps = train_maps[:200]
        train_maps = train_maps[200:]

        print("train size:{}, val size:{}".format(len(train_maps), len(val_maps)))

    # Create training generator
    train_datagen = AugmentingDataGenerator(mask_maps = masks_all, rescale=1./255)
    train_generator = train_datagen.flow(
        train_maps, 
        batch_size=batch_size,
        seed=42)
    
    # Create validation generator
    val_datagen = AugmentingDataGenerator(mask_maps = masks_all, rescale=1./255)
    val_generator = val_datagen.flow(
        val_maps, 
        batch_size=batch_size, 
        seed=42)
    
    
    return train_generator, val_generator, len(train_maps), len(val_maps)


