# Training hyperparameters
INPUTSIZE = 92
OUTPUTSIZE = (110, 110)
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = -1 # Set to -1 to train indefinitely

# Dataset
DATA_DIR = '/home/orlandi/devel/Tomography/tomo-rfx/cnn_draft/data/'
FILE_NAME = 'data_clean.npy'
NUM_WORKERS = 7

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'