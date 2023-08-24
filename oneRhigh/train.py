from model import *
from data_utils import *
from gan_utils import *
from metrics_utils import *
from model_utils import *
import numpy as np
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
sys.path.append('../')
import math


def train(G_LR, D_LR, R_LR, EPOCHS, D_STEPS, BATCH_SIZE, STEPS_PER_EPOCH, START_SIZE, END_SIZE,  cbk, pgan, aug_meta_data):

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)
    regressor_optimizer = tf.keras.optimizers.Adam(learning_rate=R_LR, beta_1=0.0, beta_2=0.999, epsilon=1e-8)

    
    pgan.compile(d_optimizer=discriminator_optimizer,g_optimizer=generator_optimizer, r_optimizer=regressor_optimizer)

    # plot_models(pgan, ARCH_ARCH_OUTPUT_PATH)

    # Start training the initial generator and discriminator
    train_dataset = CustomDataGen(aug_meta_data, X_col='id', y_col='a', rot_col = 'rotation', batch_size = BATCH_SIZE[0], 
                                  target_size=(START_SIZE, START_SIZE), freqs = [230, 345], blur = 0, shuffle=True)
    # 4 x 4
    print('SIZE: ', START_SIZE)
    history_init  = pgan.fit(train_dataset, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, callbacks=[cbk], verbose=1)
    # np.save(f'{LOSS_OUTPUT_PATH}/history_init.npy',history_init.history)

    n_depth_init = 1

    # Train faded-in/stabilized generators and discriminators
    for n_depth in range(n_depth_init, int(math.log(END_SIZE,2))-1):
        print('------------------------------------------------------------------------ \n')
        print('------------------------------------------------------------------------')

        print('SIZE: ', 4*(2**n_depth))

        pgan.n_depth = n_depth

        steps_per_epoch = STEPS_PER_EPOCH # 110 

        epochs = int(EPOCHS*(BATCH_SIZE[0]/BATCH_SIZE[n_depth])) 

        train_dataset = CustomDataGen(aug_meta_data, X_col='id', y_col='a', rot_col = 'rotation', batch_size = BATCH_SIZE[n_depth], 
                                      target_size=(4*(2**n_depth), 4*(2**n_depth)),freqs = [230, 345], blur = 0, shuffle=True)

        cbk.set_prefix(prefix=f'{n_depth}_fade_in')
        cbk.set_steps(steps_per_epoch=steps_per_epoch, epochs=epochs) 

        # Put fade in generator and discriminator
        print(f'Fading in for {(4*(2**n_depth), 4*(2**n_depth))} image..')
        pgan.fade_in_generator()
        pgan.fade_in_discriminator()
        pgan.fade_in_regressor()

        # plot_models(pgan, ARCH_ARCH_OUTPUT_PATH)

        pgan.compile(d_optimizer=discriminator_optimizer,g_optimizer=generator_optimizer, r_optimizer=regressor_optimizer)
        # Train fade in generator and discriminator
        history_fade_in = pgan.fit(train_dataset, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks=[cbk], verbose=1) 
        # np.save(f'{LOSS_OUTPUT_PATH}/history_fade_in_{n_depth}.npy',history_fade_in.history)


        # Change to stabilized generator and discriminator
        cbk.set_prefix(prefix=f'{n_depth}_stabilize')
        print(f'Stabilizing for {(4*(2**n_depth), 4*(2**n_depth))} image..')
        pgan.stabilize_generator()
        pgan.stabilize_discriminator()
        pgan.stabilize_regressor()

        # plot_models(pgan, ARCH_ARCH_OUTPUT_PATH)

        pgan.compile(d_optimizer=discriminator_optimizer,g_optimizer=generator_optimizer, r_optimizer=regressor_optimizer)

        # Train stabilized generator and discriminator
        history_stabilize = pgan.fit(train_dataset, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks=[cbk], verbose=1) #train alpha = 1 
        # np.save(f'{LOSS_OUTPUT_PATH}/history_stabilize_{n_depth}.npy',history_stabilize.history)
        
    return pgan
