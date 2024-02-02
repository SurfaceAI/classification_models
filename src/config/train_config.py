from torch import optim
from src import constants

sweep_method = {'method': 'bayes'}

sweep_metric_loss = {'metric': {'name': 'eval/loss', 'goal': 'minimize'}}

sweep_metric_acc = {'metric': {'name': 'eval/acc', 'goal': 'maximize'}}

sweep_params = {
    'batch_size': {'values': [8, 24, 48]},
    'epochs': {'values': [1, 2]},
    'learning_rate': {'distribution': 'log_uniform_values',
                      'min': 1e-05,
                      'max': 0.001},
    'optimizer_cls': {'value': constants.OPTI_ADAM},
    'augment': {'value': constants.AUGMENT_TRUE},
}

fixed_params = {
    'batch_size': 8, # 48
    'epochs': 5,
    'learning_rate': 0.001,
    'optimizer_cls': constants.OPTI_ADAM,
    'augment': constants.AUGMENT_TRUE,
}