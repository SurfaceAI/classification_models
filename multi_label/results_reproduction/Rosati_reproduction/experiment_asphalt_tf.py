import sys
sys.path.append('.')
sys.path.append('..')

from multi_label.results_reproduction.Rosati_reproduction.config_and_run import run, cfg


if __name__ == "__main__":
    seed, base_path, model_name, use_wandb, img_shape, trainable_convs, shared_layers, optimiser_params, loss_config, loss_config2, clm, obd, augment, results_path = cfg()
    run(seed, base_path, model_name, use_wandb, img_shape, trainable_convs, shared_layers, optimiser_params, loss_config, loss_config2, clm, obd, augment, results_path)
        
