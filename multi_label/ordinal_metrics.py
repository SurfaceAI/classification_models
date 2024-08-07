import torch
from sklearn.metrics import confusion_matrix

def compute_sensitivities(y_true, y_pred, labels=None):
    if len(y_true.shape) > 1:
        y_true = torch.argmax(y_true, dim=1)
    if len(y_pred.shape) > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    
    conf_mat = torch.tensor(confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy(), labels=labels), dtype=torch.float32)
    
    sum_per_class = torch.sum(conf_mat, dim=1)
    correct_per_class = torch.diag(conf_mat)
    sensitivities = correct_per_class / sum_per_class
    
    sensitivities = sensitivities[~torch.isnan(sensitivities)]
    
    return sensitivities

def minimum_sensitivity(y_true, y_pred, labels=None):
    return torch.min(compute_sensitivities(y_true, y_pred, labels=labels)).item()

def accuracy_off1(y_true, y_pred, num_classes):
    """
    Calculate 1-off accuracy.
    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        num_classes (int): Number of classes.
    Returns:
        float: 1-off accuracy.
    """
    if len(y_true.shape) > 1:
        y_true = torch.argmax(y_true, dim=1)
    if len(y_pred.shape) > 1:
        y_pred = torch.argmax(y_pred, dim=1)
    
    correct = 0
    for i in range(y_pred.size(0)):
        pred = y_pred[i]
        label = y_true[i]
        if pred == label:
            correct += 1
        elif (pred == label + 1) or (pred == label - 1):
            if 0 <= pred < num_classes:
                correct += 1
    return correct / y_pred.size(0)