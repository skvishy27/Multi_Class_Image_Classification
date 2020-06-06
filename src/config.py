import os

DATA_PATH = '../input/'
MODEL_PATH = '../models/'

TRAIN_DIR = os.path.join(DATA_PATH, 'rps')
VALID_DIR = os.path.join(DATA_PATH, 'rps-test-set')
PRED_DIR = os.path.join(DATA_PATH, 'rps-validation')

IMG_WIDTH = 150
IMG_HEIGHT = 150
IMG_CHANNELS = 3
TARGET_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32
CLASS_MODE = 'categorical'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
