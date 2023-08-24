import re
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
import pandas as pd
import tensorflow as tf

meta_data = pd.DataFrame()

def create_gif(path):
    image_folder = os.fsencode(path)

    filenames = []

    for file in os.listdir(image_folder):
        filename = os.fsdecode(file)
        if filename.endswith( ('.jpeg', '.png', '.gif') ):
            filenames.append(filename)

    filenames.sort() # this iteration technique has no built in order, so sort the frames
    images = list(map(lambda filename: imageio.v2.imread(f'{path}/'+filename), filenames))

    return filenames,images

def load_epoch_weights(PGAN, latent_dim, ckpt_epoch_path):
    xgan = PGAN(latent_dim = latent_dim)


    for n_depth in range(1,6):
        xgan.n_depth = n_depth
        xgan.fade_in_generator()
        xgan.fade_in_discriminator()
        xgan.fade_in_regressor()

        xgan.stabilize_generator()
        xgan.stabilize_discriminator()
        xgan.stabilize_regressor()


    xgan.load_weights(ckpt_epoch_path)

    return xgan

def generate_image(a_spin, pgan, num_imgs, noise_dim, typeof = 'fake', real_df = meta_data):
                                            
    if typeof == 'fake':
        random_latent_vectors = tf.random.normal(shape=[num_imgs, noise_dim])
        a_spin = np.ones([num_imgs, 1]) * a_spin

        generated_imgs = pgan.generator([random_latent_vectors, a_spin])
        fig, axes = plt.subplots(1,num_imgs,figsize=(25, 2))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_imgs[i, :, :, 1], cmap='afmhot')
            ax.axis('off')
        
        fig.suptitle(f'Fake Images with a: {a_spin[0][0]} (345 Ghz)')
        plt.show()

        
    if typeof == 'real':
        spin_df = real_df[real_df['a']==a_spin]
        data_point = spin_df.iloc[np.random.randint(low = 0, high = spin_df.shape[0], size=num_imgs)]
        path = data_point['id']
        spin = data_point['a']
        fig, axes = plt.subplots(1,num_imgs,figsize=(25, 2))
        ax = axes.flatten()
        for i,p in enumerate(path):
            image_arr = np.load('../data/' +p+'_' + 'blur{0:0=3d}'.format(0) +'.npy').astype('float32')
            image_arr = (image_arr - np.mean(image_arr))/(image_arr + np.mean(image_arr))
            image_arr = image_arr.reshape((160, 160, 1)).astype('float32')  
            image_arr = tf.image.resize(image_arr,(128, 128)).numpy() 
            ax[i].imshow(image_arr[:, :, 0], cmap='afmhot')
            ax[i].axis('off')
        fig.suptitle(f'Real Images with a: {a_spin} (345 Ghz)')
        
def generate_image_multitask(a_spin, r_high, pgan, CustomDataGen, num_imgs, noise_dim, typeof = 'fake', plot_wandb=False, real_df = meta_data):
                                            
    if typeof == 'fake':
        random_latent_vectors = tf.random.normal(shape=[num_imgs, noise_dim])
        a_spin = np.ones([num_imgs, 1]) * a_spin
        r_high = np.log2(r_high)/ np.log2(160)
        print('Normalized r_high:', r_high)
        r_high = np.ones([num_imgs,1]) * r_high
        generated_imgs = pgan.generator([random_latent_vectors, a_spin, r_high])
        fig, axes = plt.subplots(1,num_imgs,figsize=(25, 2))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated_imgs[i, :, :, 1], cmap='afmhot')
            ax.axis('off')
        
        fig.suptitle(f'Fake Images with a: {a_spin[0][0]}, R_high: {r_high[0][0]} (345 GHz)')
        plt.show()
        
        if plot_wandb:
            log_dict = {
            "Generated Images while evaluating (345 GHz)": [wandb.Image(generated_imgs[i, :, :, 1], caption='a = ' + str(a_spin[i][0])+', R_high = ' + str(r_high[i][0])) for i in range(num_imgs)]}
            
            wandb.log(log_dict)
        
    if typeof == 'real':
        filtered_df = real_df[real_df['a']==a_spin]
        filtered_df = filtered_df[filtered_df['R_high']==r_high] # int(2**(r_high * np.log2(160)))
        filtered_ds = CustomDataGen(filtered_df, X_col='id', y_col=['a','R_high'], rot_col = 'rotation', batch_size = 10, target_size=(128,128), 
                              freqs = [230,345], blur = 0, shuffle=True)

        print('Normalized r_high:', np.log2(r_high)/ np.log2(160))


        fig, axes = plt.subplots(1,num_imgs,figsize=(25, 2))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(filtered_ds[np.random.randint(0, len(filtered_ds)-1)][0][i,:,:,1], cmap='afmhot') # first batch, X, image
            ax.axis('off')

        fig.suptitle(f'Real Images with a: {a_spin}, R_high: {r_high} (345 GHz)')
        plt.show()
        
def compare_images(real_df, a_spin, pgan, noise_dim):

    
    fig, ax = plt.subplots(1,2,figsize=(10, 10))
    ax = ax.flatten()
    
    # Real Image
    spin_df = real_df[real_df['a']==a_spin]
    data_point = spin_df.iloc[np.random.randint(low = 0, high = spin_df.shape[0])]
    path = data_point['id']
    spin = data_point['a']
    image_arr = np.load('../data/' +path+'_' + 'blur{0:0=3d}'.format(0) +'.npy').astype('float32')
    image_arr = (image_arr - np.mean(image_arr))/(image_arr + np.mean(image_arr))
    image_arr = image_arr.reshape((160, 160, 1)).astype('float32')  
    image_arr = tf.image.resize(image_arr,(128, 128)).numpy() 
    ax[0].imshow(image_arr, cmap='afmhot')
    ax[0].set_title(f'Real Image - a: {spin}')
    ax[0].axis('off')
    
    # Fake Image
    random_latent_vectors = tf.random.normal(shape=[1, noise_dim])
    a_spin = np.ones([1, 1]) * a_spin
    generated_imgs = pgan.generator([random_latent_vectors, a_spin])
    ax[1].imshow(generated_imgs[0, :, :, 0], cmap='afmhot')
    ax[1].set_title(f'Fake Image - a: {a_spin[0][0]}')
    ax[1].axis('off')
    
    
    plt.show()
        
def plot_loss(loss_path):
    num_files = (len(os.listdir(loss_path)) + 1)
    fig, ax = plt.subplots(num_files//2,2, figsize=(15,25))
    ax = ax.flatten()
    i = 0
    s={}
    color = ['b','g','r','y']
    for file in os.listdir(loss_path):
        name = re.split('_|\.',file)[1]
        iteration = re.split('_|\.',file)[-2]
        if name in ['init', 'stabilize']:
            s[name + iteration] = np.load(loss_path+'/'+file,allow_pickle=True)
    s = sorted(s.items())
    for j in range(len(s)):
        ax[i].plot(s[j][1].item()['d_loss'], '.-')
        ax[i].plot(s[j][1].item()['g_loss'], '.-')

        ax[i+1].plot(s[j][1].item()['r_loss'], '.-')

        try:
            IMG_SIZE = 2**(2+j)
            ax[i].set_title(f"Image Size: {IMG_SIZE} x {IMG_SIZE}")
            ax[i+1].set_title(f"Image Size: {IMG_SIZE} x {IMG_SIZE}")
        except:
            ax[i].set_title(f"Image Size: {START_SIZE} x {START_SIZE}")
            ax[i+1].set_title(f"Image Size: {START_SIZE} x {START_SIZE}")
        ax[i].legend(['Discriminator Loss', 'Generator Loss'])
        ax[i+1].legend(['Generated Spin Loss', 'Real Spin Loss'])

        i = i + 2
