import tensorflow as tf
import numpy as np
from keras import backend as K

# Promoting variation
class MinibatchStdev(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)
    
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        average_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
        combined = tf.concat([inputs, minibatch_stddev], axis=-1)        
        return combined
    
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)

    
    
# Weight normalization    
class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.math.rsqrt(mean_square + 1.0e-8)
        normalized = inputs * l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape
    
# ELR
# https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
# stddev = sqrt(2 / fan_in)
class WeightScaling(tf.keras.layers.Layer):
    def __init__(self, shape, gain = np.sqrt(2), **kwargs):
        super(WeightScaling, self).__init__(**kwargs)
        shape = np.asarray(shape)
        shape = tf.constant(shape, dtype=tf.float32)
        fan_in = tf.math.reduce_prod(shape)
        self.wscale = gain*tf.math.rsqrt(fan_in)
      
    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs * self.wscale
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    
    
# Bias as 0 initiaalization
class Bias(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(name = "bias_logit", initial_value = b_init(shape=(input_shape[-1],), dtype='float32'), trainable=True)  

    def call(self, inputs, **kwargs):
        return inputs + self.bias
    
    def compute_output_shape(self, input_shape):
        return input_shape  
    
class WeightedSum(tf.keras.layers.Add):
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')
    
    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))
        return output

def WeightScalingDense(x, filters, gain, use_pixelnorm=False, activate=None):
    init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    in_filters = K.int_shape(x)[-1]
    x = tf.keras.layers.Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate=='LeakyReLU':
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    elif activate=='tanh':
        x = tf.keras.layers.Activation('tanh')(x)

    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x

def WeightScalingConv(x, filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=(1,1)):
    init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    in_filters = K.int_shape(x)[-1]
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate=='LeakyReLU':
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    elif activate=='tanh':
        x = tf.keras.layers.Activation('tanh')(x)
    
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x 

def RegressorConv(x, filters, kernel_size, pooling=None, activate=None, strides=(1,1)):

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", dtype='float32')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activate=='LeakyReLU':
        x = tf.keras.layers.LeakyReLU(0.01)(x)
    if pooling=='max':
        x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2)(x)
    elif pooling=='avg':
        x = tf.keras.layers.AveragePooling2D(pool_size = 4, strides = 1)(x)
    return x 
