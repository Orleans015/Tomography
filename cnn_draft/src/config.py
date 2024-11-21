# Training hyperparameters
INPUTSIZE = 92
OUTPUTSIZE = (110, 110)
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Dataset
DATA_DIR = '../data/'
FILE_NAME = 'data_clean.npy'
NUM_WORKERS = 8

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'