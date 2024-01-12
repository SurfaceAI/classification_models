from pathlib import Path
from utils import constants
gpu_kernel = 1
wandb_record = True

ROOT_DIR = Path(__file__).parent.parent
training_data_path = ROOT_DIR / 'data' / 'training'
save_path = ROOT_DIR / 'models'


