import os
import torch
import numpy as np

# from https://gist.github.com/amaarora/a2d88bfa971ce89aa5a13e006a7c94e5

class CheckpointSaver:
    def __init__(self, dirpath, saving_name, decreasing=True, top_n=5, early_stop_thresh=np.Inf):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.saving_name = saving_name
        self.top_n = top_n 
        self.decreasing = decreasing
        self.early_stop_thresh = early_stop_thresh
        self.early_stop = False
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, epoch, metric_val, optimizer=None):
        saving_name = self.saving_name.split('.',1)
        model_path = os.path.join(self.dirpath, saving_name[0] + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            # logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            data = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'metric_val': metric_val,
            }
            if optimizer is not None:
                data['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(data, model_path)
            self.top_model_paths.append({'path': model_path, 'score': metric_val, 'epoch': epoch})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        elif len(self.top_model_paths) > 0 and (epoch - self.top_model_paths[0]['epoch']) > self.early_stop_thresh:
            self.early_stop = True
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()

        return self.early_stop
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        # logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]