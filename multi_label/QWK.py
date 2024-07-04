import torch
import torch.nn as nn

def make_cost_matrix(num_ratings):
    """
    Create a quadratic cost matrix of num_ratings x num_ratings elements.

    :param num_ratings: number of ratings (classes).
    :return: cost matrix.
    """
    cost_matrix = torch.arange(num_ratings).repeat(num_ratings, 1)
    cost_matrix = (cost_matrix - cost_matrix.T).float() ** 2 / (num_ratings - 1) ** 2.0
    return cost_matrix

# Example usage:
num_ratings = 4
cost_matrix = make_cost_matrix(num_ratings)
print(cost_matrix)

def qwk_loss_base(cost_matrix):
    """
    Compute QWK loss function.
    :param cost_matrix: cost matrix.
    :return: QWK loss function.
    """
    def _qwk_loss_base(true_prob, pred_prob):
        targets = torch.argmax(true_prob, dim=1)
        costs = cost_matrix[targets]

        numerator = costs * pred_prob
        numerator = torch.sum(numerator)

        sum_prob = torch.sum(pred_prob, dim=0)
        n = torch.sum(true_prob, dim=0)

        a = torch.matmul(cost_matrix, sum_prob.view(-1, 1)).view(-1)
        b = n / torch.sum(n)

        epsilon = 1e-9

        denominator = a * b
        denominator = torch.sum(denominator) + epsilon

        loss = numerator / denominator

        return loss * 1000
    
    return _qwk_loss_base

def qwk_loss(cost_matrix, num_classes):
    """
    Compute QWK loss function.

    :param cost_matrix: cost matrix.
    :param num_classes: number of classes.
    :return: QWK loss value.
    """
    def _qwk_loss(true_labels, pred_prob):
        # Convert true_labels to one-hot encoding
        true_prob = torch.nn.functional.one_hot(true_labels, num_classes).float()
        
        # Compute the costs for the targets
        targets = torch.argmax(true_prob, dim=1)
        costs = cost_matrix[targets]

        # Compute the numerator
        numerator = costs * pred_prob
        numerator = torch.sum(numerator)

        # Compute sum_prob and n
        sum_prob = torch.sum(pred_prob, dim=0)
        n = torch.sum(true_prob, dim=0)

        # Compute the denominator
        a = torch.matmul(cost_matrix, sum_prob.view(-1, 1)).view(-1)
        b = n / torch.sum(n)

        epsilon = 1e-9

        denominator = a * b
        denominator = torch.sum(denominator) + epsilon

        return numerator / denominator

    return _qwk_loss