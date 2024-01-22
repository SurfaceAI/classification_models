from pathlib import Path
from utils import constants
gpu_kernel = 1
wandb_record = True

ROOT_DIR = Path(__file__).parent.parent
training_data_path = ROOT_DIR / 'data' / 'training'
test_data_path = ROOT_DIR / 'data' / 'testing'
data_path = ROOT_DIR / 'data'
save_path = ROOT_DIR / 'models'
