import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 256
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
OPTIMIZER_PARAMS = {'type': 'Adam', 'lr': 0.005, 'betas': (0.9, 0.999)}
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [25, 35], 'gamma': 0.2}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4

# Mixup/CutMix
IMAGE_USE_MIXUP     = False
IMAGE_MIXUP_ALPHA   = 0.2
IMAGE_USE_CUTMIX    = False
CUTMIX_ALPHA        = 1.0

# RandAugment
USE_RANDAUGMENT     = False
RANDAUGMENT_N       = 2
RANDAUGMENT_M       = 9

IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]


# Network
MODEL_NAME         = 'densenet121'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'

# 반복 학습
# OPTIMIZER_LIST = [
#     {'type': 'Adam', 'lr': 0.005, 'betas': (0.9,0.999)},
#     {'type': 'Adam', 'lr': 0.005, 'betas': (0.9,0.999)},
#     {'type': 'Adam', 'lr': 0.005, 'betas': (0.9,0.999)},
# ]

# SCHEDULER_LIST = [
#     {'type': 'CosineAnnealingLR','T_max':NUM_EPOCHS, 'eta_min':0.0},
#     {'type': 'MultiStepLR','milestones':[25,35], 'gamma':0.2},
#     {'type': 'MultiStepLR','milestones':[25,30,35], 'gamma':0.2},
# ]

# MODEL_NAME_LIST = [
#     'resnet18',
#     'resnet18',
#     'resnet18',
# ]