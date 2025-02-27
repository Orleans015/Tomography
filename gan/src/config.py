import os
import torch

config = {
    # Training hyperparameters
    'inputsize': 92,
    'img_size': (1, 110, 110), # (channels, height, width)
    'learning_rate': 3e-4,
    'batch_size': 32,
    'num_epochs': -1,
    # Dataset
    'data_dir': '/home/orlandi/devel/Tomography/tomo-rfx/gan/data/',
    'file_name': 'data_clean.npy',
    'num_workers': int(os.cpu_count()/2),
    # Compute related
    'accelerator': 'gpu',
    'devices': [0],
    # if gpu is available use it, otherwise use the cpu
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'precision': '16-mixed',
    'log_dir': '/home/orlandi/devel/Tomography/tomo-rfx/gan/logs/',
}