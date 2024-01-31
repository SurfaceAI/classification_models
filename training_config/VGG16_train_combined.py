import sys
sys.path.append('.')
sys.path.append('..')



from torch import nn, optim
from model_config import vgg16_model
from utils import training, constants

surfaces=[
                constants.ASPHALT,
                constants.CONCRETE,
                constants.PAVING_STONES,
                constants.SETT,
                constants.UNPAVED,
            ]

#habe mich entschieden über die config auch zu loopen, da ich so die Modelle direkt unter dem richtigen Namen abspeichern kann usw. 
for type_class in surfaces:
    config = dict(
        project = "road-surface-classification-quality",
        name = f"VGG16_{type_class}",
        save_name = f'VGG16_quality_{type_class}',
        architecture = "VGG16",
        dataset = 'V4', #'annotated_images',
        type_class = type_class,
        #dataset_class = 'FlattenFolders', #'FlattenFolders', #'PartialImageFolder'
        label_type = 'annotated', #'predicted
        batch_size = 16,
        valid_batch_size = 48,
        epochs = 2,
        learning_rate = 0.0002,
        seed = 42,
        validation_size = 0.2,
        image_size_h_w = (256, 256),
        crop = 'lower_middle_third',
        normalization = 'imagenet', # None, # 'imagenet', 'from_data'
        selected_classes=[
                constants.ASPHALT,
                constants.CONCRETE,
                constants.PAVING_STONES,
                constants.SETT,
                constants.UNPAVED,
            ],
        selected_quality_classes = {
                constants.ASPHALT: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                constants.CONCRETE: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                constants.PAVING_STONES: [constants.EXCELLENT, constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                constants.SETT: [constants.GOOD, constants.INTERMEDIATE, constants.BAD],
                constants.UNPAVED: [constants.INTERMEDIATE, constants.BAD, constants.VERY_BAD],
            }

    )

    augmentation = dict(
        random_horizontal_flip = True,
        random_rotation = 10,
    )

    # TODO: optimize only last layer? how to define parameters to optimize?
    optimizer = optim.Adam

    # loss, reduction='sum'
    criterion = nn.CrossEntropyLoss()

    config['selected_classes'] = config['selected_quality_classes'][type_class]
    training.config_and_train_model(config, vgg16_model.CustomVGG16, optimizer, criterion, type_class=type_class, augmentation=augmentation)