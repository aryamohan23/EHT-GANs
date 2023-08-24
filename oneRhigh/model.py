from train import *
from gan_utils import *
from data_utils import *
from gan_utils import *
from model_utils import *
import wandb
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
tf.test.gpu_device_name()
wandb.login()
# os.environ["WANDB_MODE"]="offline"



# Parameters
RHIGH = 40
BATCH_SIZE = [16, 16, 16, 16, 16, 16]
FILTERS = [512, 256, 128, 64, 32, 16]
REGRESSOR_FILTERS = [50, 50, 50, 50, 20, 10]
REGRESSOR_FILTERS_2 = [50, 50, 50, 20, 10, 10]
GP_WEIGHT = 10,
DRIFT_WEIGHT=0.001


class PGAN(Model):
    def __init__(self, latent_dim, d_steps, gp_weight = GP_WEIGHT, drift_weight = DRIFT_WEIGHT):
        super(PGAN, self).__init__()
        self.latent_dim = latent_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.drift_weight = drift_weight
        self.n_depth = 0
        self.discriminator = self.init_discriminator()
        self.discriminator_wt_fade = None
        self.generator = self.init_generator()
        self.regressor = self.init_regressor()
        self.generator_wt_fade = None

    def call(self, inputs):
        return

    def init_discriminator(self):
        img_input = tf.keras.layers.Input(shape = (4,4,2))
        img_input = tf.cast(img_input, tf.float32)

        # fromGrayScale
        x = WeightScalingConv(img_input, filters = FILTERS[0], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU') # 4 x 4 x 512
        
        # Add Minibatch end of discriminator
        x = MinibatchStdev()(x) # 4 x 4 x 513

        x = WeightScalingConv(x, filters = FILTERS[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 4 x 4 x 512
        
        x = WeightScalingConv(x, filters = FILTERS[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', strides=(4,4)) # 1 x 1 x 512

        x = tf.keras.layers.Flatten()(x)
        
        x = WeightScalingDense(x, filters=1, gain=1.)

        d_model = Model(img_input, x, name='discriminator')

        return d_model

    # Fade in upper resolution block
    def fade_in_discriminator(self):

        input_shape = list(self.discriminator.input.shape) 
        # 1. Double the input resolution. 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3]) # 8 x 8 x 2
        img_input = tf.keras.layers.Input(shape = input_shape)
        img_input = tf.cast(img_input, tf.float32)

        # 2. Add pooling layer 
        #    Reuse the existing “FromGrayScale” block defined as “x1" -- SKIP CONNECTION (ALREADY STABILIZED -> 1-alpha)
        x1 = tf.keras.layers.AveragePooling2D()(img_input) # 4 x 4 x 1
        x1 = self.discriminator.layers[1](x1) # Conv2D FromGrayScale # 4 x 4 x 512
        x1 = self.discriminator.layers[2](x1) # WeightScalingLayer # 4 x 4 x 512
        x1 = self.discriminator.layers[3](x1) # Bias # 4 x 4 x 512
        x1 = self.discriminator.layers[4](x1) # LeakyReLU # 4 x 4 x 512

        # 3.  Define a "fade in" block (x2) with a new "fromGrayScale" and two 3x3 convolutions.
        # symmetric
        x2 = WeightScalingConv(img_input, filters = FILTERS[self.n_depth], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 256

        x2 = WeightScalingConv(x2, filters = FILTERS[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 256
        x2 = WeightScalingConv(x2, filters = FILTERS[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU') # 8 x 8 x 512

        x2 = tf.keras.layers.AveragePooling2D()(x2) # 4 x 4 x 512

        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator 
        for i in range(5, len(self.discriminator.layers)):
            x2 = self.discriminator.layers[i](x2)
        self.discriminator_stabilize = Model(img_input, x2, name='discriminator')

        # 5. Add existing discriminator layers. 
        for i in range(5, len(self.discriminator.layers)):
            x = self.discriminator.layers[i](x)
        self.discriminator = Model(img_input, x, name='discriminator')

    # Change to stabilized(c. state) discriminator 
    def stabilize_discriminator(self):
        self.discriminator = self.discriminator_stabilize
        
    def init_regressor(self):
        
        img_input = tf.keras.layers.Input(shape = (4, 4, 2))
        img_input = tf.cast(img_input, tf.float32)
                
        #  [(I - F +2 *P) / S] +1 = 4 x 4 x 50

        x = RegressorConv(img_input, REGRESSOR_FILTERS[0], kernel_size = 1, pooling=None, activate='LeakyReLU', strides=(1,1))
        
        
        # print(x.shape) # 4 x 4 x 50
        x = RegressorConv(x, REGRESSOR_FILTERS[0], kernel_size = 3, pooling='avg', activate='LeakyReLU', strides=(1,1)) 
        # print(x.shape) # should be 1 x 1 x 50
        x = tf.keras.layers.Flatten()(x) # 50
        x = tf.keras.layers.Dense(units = 16)(x) # 16
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        
        x = tf.keras.layers.Dense(units = 1)(x) # 1

        c_model = Model(img_input, x, name='regressor')

        return c_model

    def fade_in_regressor(self):

        input_shape = list(self.regressor.input.shape)
        
        # 1. Double the input resolution. 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3]) # 8 x 8 x 2
        img_input = tf.keras.layers.Input(shape = input_shape)
        img_input = tf.cast(img_input, tf.float32)

        # 2. Add pooling layer 
        x1 = tf.keras.layers.AveragePooling2D()(img_input) 
        x1 = self.regressor.layers[1](x1) # Conv2D 
        x1 = self.regressor.layers[2](x1) # BatchNormalization 
        x1 = self.regressor.layers[3](x1) # LeakyReLU 

        # 3.  Define a "fade in" block (x2) with a new "fromGrayScale" and two 3x3 convolutions.
        
        if self.n_depth!=5:
            x2 = RegressorConv(img_input, REGRESSOR_FILTERS_2[self.n_depth], kernel_size = 1, pooling=None, activate='LeakyReLU', strides=(1,1))

            x2 = RegressorConv(x2, REGRESSOR_FILTERS[self.n_depth], kernel_size = 3, pooling='max', activate='LeakyReLU', strides=(1,1))
            
        else:
            x2 = RegressorConv(img_input, REGRESSOR_FILTERS[self.n_depth], kernel_size = 3, pooling='max', activate='LeakyReLU', strides=(1,1))

        
        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator 
        for i in range(4, len(self.regressor.layers)):
            x2 = self.regressor.layers[i](x2)
        self.regressor_stabilize = Model(img_input, x2, name='regressor')

        # 5. Add existing discriminator layers. 
        for i in range(4, len(self.regressor.layers)):
            x = self.regressor.layers[i](x)
        self.regressor = Model(img_input, x, name='regressor')

    # Change to stabilized(c. state) discriminator 
    def stabilize_regressor(self):
        self.regressor = self.regressor_stabilize

    def init_generator(self):
        noise = tf.keras.layers.Input(shape=(self.latent_dim,)) # None, 512
        a_spin = tf.keras.layers.Input(shape=(1,))
        
        merge = tf.keras.layers.Concatenate()([noise, a_spin]) #L x (3)
                
        # Actual size(After doing reshape) is just FILTERS[0], so divide gain by 4
 
        x = WeightScalingDense(merge, filters=4*4*FILTERS[0], gain=np.sqrt(2)/4, activate='LeakyReLU', use_pixelnorm=False) 
        
        x = tf.keras.layers.Reshape((4, 4, FILTERS[0]))(x)

        x = WeightScalingConv(x, filters = FILTERS[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=False)
        
        x = WeightScalingConv(x, filters = FILTERS[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)

        # Gain should be 1 as its the last layer 
        x = WeightScalingConv(x, filters=2, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False) # change to tanh and understand gain 1 if training unstable

        g_model = Model([noise,a_spin], x, name='generator')

        return g_model

    # Fade in upper resolution block
    def fade_in_generator(self):

        # 1. Get the node above the “toGrayScale” block 
        block_end = self.generator.layers[-5].output
        
        # 2. Upsample block_end       
        block_end = tf.keras.layers.UpSampling2D((2,2))(block_end) # 8 x 8 x 512

        # 3. Reuse the existing “toGrayScale” block defined as“x1”. --- SKIP CONNECTION (ALREADY STABILIZED)
        x1 = self.generator.layers[-4](block_end) # Conv2d
        x1 = self.generator.layers[-3](x1) # WeightScalingLayer
        x1 = self.generator.layers[-2](x1) # Bias
        x1 = self.generator.layers[-1](x1) # tanh

        # 4. Define a "fade in" block (x2) with two 3x3 convolutions and a new "toRGB".
        x2 = WeightScalingConv(block_end, filters = FILTERS[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True) # 8 x 8 x 512 
        
        x2 = WeightScalingConv(x2, filters = FILTERS[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True) # 8 x 8 x 512 
        
        # "toGrayScale"
        x2 = WeightScalingConv(x2, filters=2, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False) # 

        # Define stabilized(c. state) generator
        self.generator_stabilize = Model(self.generator.input, x2, name='generator')

        # 5.Then "WeightedSum" x1 and x2 to smoothly put the "fade in" block.
        x = WeightedSum()([x1, x2])
        self.generator = Model(self.generator.input, x, name='generator')

    # Change to stabilized(c. state) generator 
    def stabilize_generator(self):
        self.generator = self.generator_stabilize

    def compile(self, d_optimizer, g_optimizer, r_optimizer):
        super(PGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.r_optimizer = r_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        diff = fake_images - real_images
        interpolated = real_images + epsilon * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        
        real_images, real_a_spin = data
        batch_size = tf.shape(real_images)[0]
        indices = tf.random.shuffle(tf.range(2 * batch_size))

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator([random_latent_vectors, real_a_spin], training=False)
            combined_images = tf.concat([generated_images, real_images], axis=0)
            shuffled_combined_images = tf.gather(combined_images, indices)
            
            with tf.GradientTape() as d_tape:

                # Watch the gradients
                d_tape.watch(self.discriminator.trainable_weights)

                # Train discriminator
                pred_logits = self.discriminator(shuffled_combined_images, training=True)
                unshuffled_pred_logits = tf.gather(pred_logits, tf.argsort(indices))  

                # Wasserstein Loss
                d_fake = tf.reduce_mean(unshuffled_pred_logits[:batch_size])
                d_real = tf.reduce_mean(unshuffled_pred_logits[batch_size:])

                d_cost = d_fake - d_real

                # Gradient Penalty
                gp = self.gradient_penalty(batch_size, real_images, generated_images)

                # Drift added by PGGAN paper
                drift = tf.reduce_mean(tf.square(pred_logits))

                # WGAN-GP
                d_loss = d_cost + (self.gp_weight * gp) + (self.drift_weight * drift) 

            d_gradient = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_weights))     
        
        with tf.GradientTape() as r_tape:
            
            # Watch the gradients
            r_tape.watch(self.regressor.trainable_weights)

            # Train regressor
            pred_a_spin = self.regressor(real_images, training=True)

            # Loss on spin 
            r_loss = tf.keras.losses.MeanAbsoluteError()(real_a_spin, pred_a_spin) 

        r_gradient = r_tape.gradient(r_loss, self.regressor.trainable_weights) 
        self.r_optimizer.apply_gradients(zip(r_gradient, self.regressor.trainable_weights))

        
        with tf.GradientTape() as g_tape:
            
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            g_tape.watch(self.generator.trainable_weights)
            
            gen = self.generator([random_latent_vectors, real_a_spin], training=True)
            predictions = self.discriminator(gen, training = False)
            predictions_a_spin = self.regressor(gen, training = False)
            
            # Total generator loss
            a_spin_loss = tf.keras.losses.MeanAbsoluteError()(real_a_spin, predictions_a_spin)
            
            g_cost = tf.reduce_mean(predictions)
            g_loss = -g_cost + a_spin_loss 
            
            
        # Get the gradients
        g_gradient = g_tape.gradient(g_loss , self.generator.trainable_weights)
        # Update the weights 
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_weights))


        
        return {'d_loss': d_loss, 'g_loss': g_loss, 'r_loss': r_loss}


def compute_tstr(pgan, d_steps):
    
    pgan_regressor = PGAN(latent_dim = NOISE_DIM, d_steps =  d_steps)
    
    xgan = PGAN(latent_dim = NOISE_DIM, d_steps =  d_steps)

    for n_depth in range(1,6):
        xgan.n_depth = n_depth
        xgan.fade_in_generator()
        xgan.fade_in_discriminator()
        xgan.fade_in_regressor()

        xgan.stabilize_generator()
        xgan.stabilize_discriminator()
        xgan.stabilize_regressor()

    tstr_regressor = xgan.regressor 

    noise_dim = NOISE_DIM
    num_imgs = 505

    random_latent_vectors = tf.random.normal(shape=[num_imgs, noise_dim])
    random_a = np.round(tf.random.uniform([num_imgs, 1], minval=-1, maxval=1),2)
    random_a += 0.

    generated_imgs = pgan.generator.predict([random_latent_vectors, random_a])  # num_images x 128 x 128 x 1
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience= 0)


    x_train, x_val, y_train, y_val = train_test_split(generated_imgs, random_a, test_size=0.3)

    real_dataset = CustomDataGen(meta_data, X_col='id', y_col='a', rot_col = None, batch_size = 505, target_size=(128,128), 
                                 freqs = [230,345], blur = 0, shuffle=True)

    tstr_regressor.compile(keras.optimizers.Adam(learning_rate=0.0005),  loss=tf.keras.losses.MeanSquaredError())

    history = tstr_regressor.fit(x_train, y_train,  validation_data=(x_val,y_val), epochs=200, batch_size=32, callbacks=[early_stop]) 

    for X,y in real_dataset:
        y_true_test = y
        X_real = X
        break
    
    y_pred_test = tstr_regressor.predict(X_real)
    
    
    r2_score = metrics.r2_score(y_true_test, y_pred_test)

    return r2_score

