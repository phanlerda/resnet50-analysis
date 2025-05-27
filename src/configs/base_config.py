# BACKUP/src/configs/base_config.py
import torch

# --- PATHS ---
# Đường dẫn đến dữ liệu training đã chia (và align nếu input đã align)
ALIGNED_DATA_DIR = "../data/dataset_splits/train" 
# Đường dẫn đến dữ liệu validation đã chia (và align nếu input đã align)
ALIGNED_VAL_DATA_DIR = "../data/dataset_splits/val" 
# Đường dẫn đến dữ liệu test đã chia (và align nếu input đã align)
ALIGNED_TEST_DATA_DIR = "../data/dataset_splits/test"

WEIGHTS_DIR = "../weights"

RESULTS_DIR = "../results"

# --- DATASET ---
IMAGE_SIZE = 224
NUM_CLASSES_VGGFACE2_FULL = 8631 # Kiểm tra lại số class trong RAW_DATA_DIR

# --- TRAINING ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
STEP_SIZE = 10
GAMMA = 0.1

# --- MODEL SPECIFIC ---
EMBEDDING_DIM = 512

# --- PREPROCESSING ---
MTCNN_IMAGE_SIZE = 160
MTCNN_MARGIN = 0
ALIGN_SAVE_MARGIN = 20
ALIGN_SAVE_IMG_SIZE = IMAGE_SIZE