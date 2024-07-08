# Training hyperparameters
INPUTSIZE = 784
OUTPUTSIZE = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Dataset
DATA_DIR = '../data/'
NUM_WORKERS = 4

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'