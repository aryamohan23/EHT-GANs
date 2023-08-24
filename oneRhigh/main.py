import sys
sys.path.append('../')
from model import *
from train import *
from gan_utils import *
from data_utils import *
from gan_utils import *
from model_utils import *
import wandb

wandb.init()

D_STEPS = 5
EPOCHS = 100
NOISE_DIM = 512
NUM_IMGS_GENERATE = 9
STEPS_PER_EPOCH = 110
START_SIZE = 4
END_SIZE = 128


aug_meta_data = load_meta_data(RHIGH)




pgan = PGAN(latent_dim = NOISE_DIM, d_steps = D_STEPS)
cbk = GANMonitor(num_img = NUM_IMGS_GENERATE, latent_dim = NOISE_DIM)
cbk.set_steps(steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS) # 110, 6
cbk.set_prefix(prefix='0_init')

# Cluster - WandB sweep
pgan = train(wandb.config.G_LR, wandb.config.D_LR, wandb.config.R_LR, EPOCHS, D_STEPS, BATCH_SIZE, STEPS_PER_EPOCH, START_SIZE, END_SIZE,  cbk, pgan, aug_meta_data)

# Local
# pgan = train(0.001, 0.001, 0.001, EPOCHS, D_STEPS, BATCH_SIZE, STEPS_PER_EPOCH, START_SIZE, END_SIZE,  cbk, pgan, aug_meta_data)

tstr = compute_tstr(pgan, D_STEPS)

wandb.log({"tstr": tstr})

