from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import seed
from copy import deepcopy
import numpy as np
import gc
import tensorflow as tf

import random
def gen_pair(pair_len, batch_size):
    id_1 = list(range(pair_len))
    id_2 = list(range(pair_len))
    random.shuffle(id_1)
    random.shuffle(id_2)
    while True:
        redo = False
        for i in range(len(id_1)):
            if id_1[i] == id_2[i]:
                redo = True
                random.shuffle(id_2)
                break
        if not redo:
            break
    id_1 = np.array(id_1)
    id_2 = np.array(id_2) + batch_size 
    return np.concatenate([id_1,id_2])

class AugmentingDataGenerator(ImageDataGenerator):
    
    def __init__(self, mask_maps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_maps = mask_maps
    
    def self_mask_generator(self, rand_seed=42):
        if rand_seed:
            seed(rand_seed)
#         ks = list(self.mask_maps.keys())
        kid = np.random.choice(range(len(self.mask_maps)))
        selected_class = self.mask_maps[kid]
#         maskid = np.random.choice(range(len(selected_class)))
        return selected_class
    
    def flow(self, x, batch_size, *args, **kwargs):
        generator = super().flow(x, batch_size=batch_size, *args, **kwargs)
#         generator2 = super().flow(x_con, orimask, batch_size=batch_size,
#                                   *args, **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']
        while True:
            
            # construct train samples for inpainting optimization 
            ori = next(generator) 
            ori_c = deepcopy(ori)
            
#             ori_v = np.flip(ori, 0)
            ori1 = np.concatenate([ori, ori], axis=0)
            mask = []
            for i in range(ori1.shape[0]):
                imgi = ori1[i]
                maski = self.self_mask_generator(seed)
                red, green, blue= imgi.T
                white_areas = (red == 0) & (blue == 0) & (green == 0)
                maski[[white_areas.T]]=(1,1,1)
#                 maski[imgi==[0,0,255]] = 1 # exclude the invalid pixels (blue ones)
                mask.append(maski)
            mask = np.array(mask)
            masked = deepcopy(ori1)
            masked[mask==0] = 1.
            
            # construct train samples for contrastive and consistency optimization
            v_ori2 = np.flip(ori_c, 1)
            v_ori2mask = []
            for i in range(v_ori2.shape[0]):
                imgi = v_ori2[i]
                maski = self.self_mask_generator(seed)
                red, green, blue= imgi.T
                white_areas = (red == 0) & (blue == 0) & (green == 0)
                maski[[white_areas.T]]=(1,1,1)
#                 maski[imgi==[0,0,255]] = 1 # exclude the invalid pixels (blue ones)
                v_ori2mask.append(maski)
            v_ori2mask = np.array(v_ori2mask)
            masked_v_ori2 = deepcopy(v_ori2)
            masked_v_ori2[v_ori2mask==0] = 1.
            
            h_ori2 = np.flip(ori_c, 2)
            h_ori2mask = []
            for i in range(h_ori2.shape[0]):
                imgi = h_ori2[i]
                maski = self.self_mask_generator(seed)
                red, green, blue= imgi.T
                white_areas = (red == 0) & (blue == 0) & (green == 0)
                maski[[white_areas.T]]=(1,1,1)
#                 maski[imgi==[0,0,255]] = 1 # exclude the invalid pixels (blue ones)
                h_ori2mask.append(maski)
            h_ori2mask = np.array(h_ori2mask)
            masked_h_ori2 = deepcopy(h_ori2)
            masked_h_ori2[h_ori2mask==0] = 1.
            
            # combine list 
            masked_cc = np.concatenate((masked_v_ori2, masked_h_ori2), axis=0)
            mask_cc = np.concatenate((v_ori2mask, h_ori2mask), axis=0)
            ori_cc = np.concatenate((v_ori2, h_ori2), axis=0)
            
            # id pairs
            pair_len = ori.shape[0]
            id_pairs = gen_pair(pair_len, batch_size)

            gc.collect()
#             yield [masked, mask, masked_cc, mask_cc, id_pairs], ori1
            yield [masked, mask, masked_cc, mask_cc, id_pairs], [ori1, ori_cc, ori_cc]
#             yield [masked, mask, masked_cc, mask_cc, id_pairs], ori1
            #yield [masked, mask, masked_cc, mask_cc, id_pairs], [ori1, ori_cc]
            #yield [masked, mask, masked_cc, mask_cc, id_pairs], ori_cc
                
