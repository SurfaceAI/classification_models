from pathlib import Path

config = {
    gpu_kernel = 1,

    ROOT_DIR = Path(__file__).parent.parent.parent,
    training_data_path = ROOT_DIR / 'data' / 'training',
    save_path = ROOT_DIR / 'models',

    project = "road-surface-classification-type",

    # dataset
    dataset = 'V4', #'annotated_images',
    label_type = 'annotated', #'predicted
    image_size_h_w = (768, 768),
    crop_size = [512, 256, 256, 256],
    norm_mean = [0.485, 0.456, 0.406],
    norm_std = [0.229, 0.224, 0.225],
    augmentation = dict(
        random_horizontal_flip = True,
        random_rotation = 10,
    ),

    seed = 42

}
