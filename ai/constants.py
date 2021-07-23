RANDOM_SEED = 5
ROOT = "../datasets"
IMG_WIDTH = 128
IMG_HEIGHT = 128
TRAIN_SIZE = 0.7

# For model training
BATCH_SIZE = 128
EPOCHS = 3
MAX_ACCURACY = 0

# Image classification distance threshold
IMG_DIST_THRESHOLD = 1.3

# Folder where uploaded images are places in the fruits folder
PREDICT_FOLDER = 'new'

# Threshold for predictions
PREDICT_THRESHOLD = 1.25

# Class Confusion matrix filename
CFM_FILENAME = "class_confusion_matrix_triplet_loss.png"
CFM_FILENAME_TRUE = "class_confusion_matrix_triplet_loss_true.png"
CFM_FILENAME_FALSE = "class_confusion_matrix_triplet_loss_false.png"

# Complete confusion matrix filename
COMPLETE_CFM_FILENAME = "comp_confusion_matrix_triplet_loss.png"

# Siamese model file name
SIAMESE_MODEL_FILE = "siamese_model"
FINAL_SIAMESE_MODEL_FILE = "siamese_model-final"

# Input shape for Xception model for transfer learning
INPUT_SHAPE = (128, 128, 3)
