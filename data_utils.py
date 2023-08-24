import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize, rotate
import tensorflow as tf

def create_folders():
    
    version='_oneRhigh'
    CKPT_OUTPUT_PATH = 'VAC-PGGAN_ckpts'+version
    IMG_OUTPUT_PATH_230 = 'VAC-PGGAN_Images_230GHz'+version
    IMG_OUTPUT_PATH_345 = 'VAC-PGGAN_Images_345GHz'+version
    ARCH_OUTPUT_PATH = 'VAC-PGGAN_Arch'+version
    LOSS_OUTPUT_PATH = 'VAC-PGGAN_Loss'+version
    DATASET_OUTPUT_PATH = 'synthetic_data'+version

    try:
        os.mkdir(CKPT_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(IMG_OUTPUT_PATH_230)
    except FileExistsError:
        pass

    try:
        os.mkdir(IMG_OUTPUT_PATH_345)
    except FileExistsError:
        pass

    try:
        os.mkdir(ARCH_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(LOSS_OUTPUT_PATH)
    except FileExistsError:
        pass

    try:
        os.mkdir(DATASET_OUTPUT_PATH)
    except FileExistsError:
        pass
    
    return CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH_230, IMG_OUTPUT_PATH_345, ARCH_OUTPUT_PATH, LOSS_OUTPUT_PATH, DATASET_OUTPUT_PATH

def augment_by_rotation(data, angles=[90,180,270]):
    original_data = data.copy()
    for angle in angles:
        original_data['rotation'] = angle
        data = pd.concat([data,original_data])
    return data

def get_unique(data):
    for col in data.columns[1:]:
        print(f'\n"{col}" has {len(data[col].unique())} unique values: {data[col].unique()}')
        
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, meta_data, X_col, y_col, rot_col, batch_size, target_size, freqs = [230, 345], blur = 0, shuffle=True):
        
        self.meta_data = meta_data.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.freqs = freqs
        self.blur = blur
        self.rot_col = rot_col
        self.n = len(self.meta_data)

    def on_epoch_end(self):
        if self.shuffle:
            print('Shuffing the data..')
            self.meta_data = self.meta_data.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, img_id, target_size, rotation_angle=0):

        imgs = {}

        for freq in self.freqs:
            file_name = '_'.join(
                    img_id.split('_')[:5] + ['freq={}'.format(freq)]
                    + img_id.split('_')[6:] + ['blur{0:0=3d}'.format(self.blur)]
                    ) +'.npy'
            
            file_name = '../data/' + file_name
            
            imgs[freq] = np.load(file_name).astype('float32')

            if self.rot_col:
                imgs[freq] = rotate(imgs[freq], rotation_angle)

            imgs[freq] = (imgs[freq] - np.mean(imgs[freq]))/(imgs[freq] + np.mean(imgs[freq]))

        stacked_img = np.stack(list(imgs.values()), axis=2) 

        image_arr = tf.image.resize(stacked_img,(target_size[0], target_size[1])).numpy()
        
        return image_arr
    
    def __get_output(self, label):
        return label
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_col_batch = batches[self.X_col]

        if self.rot_col:
            rot_col_batch = batches[self.rot_col]
            X_batch = np.asarray([self.__get_input(x, self.target_size, rot) for (x,rot) in zip(X_col_batch,rot_col_batch)])
        else: 
            X_batch = np.asarray([self.__get_input(x, self.target_size) for x in X_col_batch])
            
        y_col_batch = batches[self.y_col]
        
        
        y_batch = np.asarray([self.__get_output(y) for y in y_col_batch])
        


        return X_batch, y_batch
    
    def __getitem__(self, index):
        
        # The role of __getitem__ method is to generate one batch of data. 
        
        meta_data_batch = self.meta_data[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(meta_data_batch)
        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

def load_meta_data(RHIGH):
    meta_data = pd.read_csv("../data.csv")
    meta_data=meta_data[meta_data['R_high']==RHIGH]

    # Getting only 230 freq because data generator stacks 345 images by default anyway
    meta_data=meta_data[meta_data['freq']==230]

    meta_data = meta_data[['id','a', 'R_high','freq' ,'frame','rotation']].drop_duplicates().sort_values(by=['a', 'R_high']).reset_index(drop=True)
    print(f"Data Shape: {meta_data.shape}")
    aug_meta_data = augment_by_rotation(meta_data)
    print(f"Data Shape of augmented dataset: {aug_meta_data.shape}")

    # Showing what all is in my data
    get_unique(aug_meta_data)
    
    return aug_meta_data
