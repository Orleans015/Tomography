# Training hyperparameters
INPUTSIZE = 92
OUTPUTSIZE = 21
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
NUM_EPOCHS = -1 # Set to -1 to train indefinitely

# Dataset
DATA_DIR = '../data/'
FILE_NAME = 'data_clean.npy'
NUM_WORKERS = 8

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'