# Training hyperparameters
INPUTSIZE = 92
OUTPUTSIZE = 21
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 100

# Dataset
DATA_DIR = '../data/'
FILE_NAME = 'data.npy'
NUM_WORKERS = 4

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'