import numpy as np

ROOT = '/home/ubuntu/project/generated GCD'
DATA_ROOT='/home/ubuntu/Datasets/'

# -----------------
# DATASET PATHS
# -----------------
cifar_10_root = f'{DATA_ROOT}'
cifar_100_root = f'{DATA_ROOT}'
cub_root = f'{DATA_ROOT}/CUB'
aircraft_root = f'{DATA_ROOT}/aircraft/'
herbarium_dataroot = f'{DATA_ROOT}/herbarium_2019/'
imagenet_root = f'{DATA_ROOT}/imagenet-img'
imagenet_gcd_root = f'{DATA_ROOT}/imagenet_100_gcd'

# -----------------
# OTHER PATHS
# -----------------
osr_split_dir = f'{ROOT}/data/ssb_splits' # OSR Split dir
feature_extract_dir = f'{ROOT}/tmp'     # Extract features to this directory
exp_root = f'{ROOT}/cache'          # All logs and checkpoints will be saved here

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/home/ubuntu/project/pretrain/dino_vitbase16_pretrain.pth'
pretrain_path = '/home/ubuntu/project/pretrain/'
ibot_pretrain_path = '/media/cs4007/adfc2692-0951-4a9b-8ea6-ebfa0e11323b/lcm/Pretraining/dino_vitbase16_pretrain.pth/checkpoint_teacher.pth'
