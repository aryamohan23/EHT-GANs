import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import wandb
from data_utils import *
from model_utils import *

    
def plot_models(pgan, ARCH_OUTPUT_PATH, typeof='init'):

    if typeof=='fade in':
        tf.keras.utils.plot_model(pgan.generator, to_file=f'{ARCH_OUTPUT_PATH}/generator_{pgan.n_depth}_fade_in.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.discriminator, to_file=f'{ARCH_OUTPUT_PATH}/discriminator_{pgan.n_depth}_fade_in.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.regressor, to_file=f'{ARCH_OUTPUT_PATH}/regressor_{pgan.n_depth}_fade_in.png', show_shapes=True)
        
    elif typeof=='stabilize':
        tf.keras.utils.plot_model(pgan.generator, to_file=f'{ARCH_OUTPUT_PATH}/generator_{pgan.n_depth}_stabilize.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.discriminator, to_file=f'{ARCH_OUTPUT_PATH}/discriminator_{pgan.n_depth}_stabilize.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.regressor, to_file=f'{ARCH_OUTPUT_PATH}/regressor_{pgan.n_depth}_stabilize.png', show_shapes=True)
        
    else:
        tf.keras.utils.plot_model(pgan.generator, to_file=f'{ARCH_OUTPUT_PATH}/generator_{pgan.n_depth}.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.discriminator, to_file=f'{ARCH_OUTPUT_PATH}/discriminator_{pgan.n_depth}.png', show_shapes=True)
        tf.keras.utils.plot_model(pgan.regressor, to_file=f'{ARCH_OUTPUT_PATH}/regressor_{pgan.n_depth}.png', show_shapes=True)
        
CKPT_OUTPUT_PATH, IMG_OUTPUT_PATH_230, IMG_OUTPUT_PATH_345, ARCH_OUTPUT_PATH, LOSS_OUTPUT_PATH, DATASET_OUTPUT_PATH = create_folders()


# Saves generated images and updates alpha in WeightedSum layers
class GANMonitor(tf.keras.callbacks.Callback):
    
    def __init__(self, num_img, latent_dim, prefix='', checkpoint_path = ''):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.random_latent_vectors = tf.random.normal(shape=[num_img, self.latent_dim])
        self.a_spin = np.round(tf.random.uniform(shape=[num_img, 1], minval=-1,maxval=1),2)
        self.steps_per_epoch = 0
        self.epochs = 0
        self.steps = self.steps_per_epoch * self.epochs
        self.n_epoch = 0
        self.prefix = prefix
        self.checkpoint_path = checkpoint_path
        self.absolute_epoch = 0
        
  
    def set_prefix(self, prefix=''):
        self.prefix = prefix
        
    def set_steps(self, steps_per_epoch, epochs):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.steps = self.steps_per_epoch * self.epochs # 660

    def on_epoch_begin(self, epoch, logs=None):
        self.n_epoch = epoch
        checkpoint_path = f"{CKPT_OUTPUT_PATH}/pgan_{self.prefix}/pgan_{self.n_epoch:05d}.ckpt"
        self.checkpoint_path = checkpoint_path

    def on_epoch_end(self, epoch, logs=None):

        prefix_number = int(self.prefix.split("_")[0])
        prefix_state = str(self.prefix.split("_")[1])

        # Plot epoch end generated images
        n_grid = int(np.sqrt(self.num_img))
        generated_imgs = self.model.generator([self.random_latent_vectors,self.a_spin])
        fig, axes = plt.subplots(n_grid,n_grid,figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_imgs[i, :, :, 1], cmap='gray') 
            img_spin = self.a_spin[i][0]
            ax.set_title(f'a: {str(img_spin)}')
            ax.axis('off')
            plt.savefig(f'{IMG_OUTPUT_PATH_345}/plot_{self.prefix}_{epoch:05d}.jpeg')
        plt.suptitle('345 GHz', fontsize = 15, fontweight = 'bold')
        plt.show()


         

        image_size = (4*(2**prefix_number), 4*(2**prefix_number))

        if (prefix_state=='stabilize') or (prefix_state=='init'):

            log_dict = {
            "Generated Images while training (345 GHz)": [wandb.Image(generated_imgs[i, :, :, 1], 
                                            caption='a = ' + str(self.a_spin[i][0])) for i in range(self.num_img)],
            f'Discriminator Loss ({image_size[0]}x{image_size[0]})': logs['d_loss'],
            f'Generator Loss ({image_size[0]}x{image_size[0]})': logs['g_loss'],
            f'Spin Loss on real data ({image_size[0]}x{image_size[0]})': logs['r_loss'], 
            f'Epoch':self.n_epoch
            }
            wandb.log(log_dict) # was commit=False and was working, but only logged last value, working without it as well

             
        if (prefix_number == 5) and (prefix_state=='stabilize') and ((epoch%2==0) or (epoch==self.epochs-1)):
            print('Saving weights...')
            self.model.save_weights(self.checkpoint_path)
            print('Successfuly saved weights.')
            
        if (prefix_number == 4) and (prefix_state=='stabilize') and ((epoch%3==0) or (epoch==self.epochs-1)):
            print('Saving weights...')
            self.model.save_weights(self.checkpoint_path)
            print('Successfuly saved weights.')
            
            
    def on_batch_begin(self, batch, logs=None):
        
        # Update alpha in WeightedSum layers
        # alpha usually goes from 0 to 1 evenly over ALL the epochs for that depth.
        alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / float(self.steps - 1) #1/219  to 1*110+109/220 for 2 epochs
        
        # print(f'!!! From GANMonitor: Steps: {self.steps}, Epoch: {self.n_epoch}, Steps per epoch: {self.steps_per_epoch}, Batch: {batch}, Alpha: {alpha}')
        
        for layer in self.model.generator.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)
        for layer in self.model.discriminator.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)



