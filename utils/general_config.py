from pathlib import Path

gpu_kernel = 1

cwd = Path.cwd()
training_data_path = cwd / 'data' / 'training'
save_path = cwd.parent / 'models'
