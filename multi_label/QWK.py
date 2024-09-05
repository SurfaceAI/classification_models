
import torch
import torch.nn as nn

class QWK_Loss(nn.Module):
    """
    Implements Weighted Kappa Loss. Weighted Kappa is widely used in Ordinal Classification Problems.
    """

    def __init__(self, num_classes: int, mode = 'quadratic', epsilon= 1e-10):
        """
        Args:
            num_classes: Number of unique classes in the dataset.
            mode: Weighting mode, either 'linear' or 'quadratic'. Defaults to 'quadratic'.
            epsilon: Small constant to avoid log of zero. Defaults to 1e-10.
        """
        super(QWK_Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        if mode == 'quadratic':
            self.y_pow = 2
        elif mode == 'linear':
            self.y_pow = 1
        else:
            raise ValueError("mode must be 'linear' or 'quadratic'")

    def kappa_loss(self, y_pred, y_true):
        """
        Computes the Quadratic Weighted Kappa loss.
        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Ground truth labels.
        """
        device = y_pred.device  # Ensuring compatibility with device (cuda/cpu)

        # One-hot encode the true labels
        y_true_one_hot = torch.eye(self.num_classes, device=device)[y_true]

        # Predicted probabilities after applying the chosen power
        pred_ = y_pred ** self.y_pow
        pred_norm = pred_ / (torch.sum(pred_, dim=1, keepdim=True) + self.epsilon)

        # Histograms of the predicted and true classes
        hist_rater_a = torch.sum(pred_norm, dim=0)
        hist_rater_b = torch.sum(y_true_one_hot, dim=0)

        # Confusion matrix between predicted and true labels
        conf_mat = torch.matmul(pred_norm.t(), y_true_one_hot)

        # Cost matrix for weighting disagreements between classes
        repeat_op = torch.arange(self.num_classes, device=device).unsqueeze(1)
        repeat_op_sq = (repeat_op - repeat_op.t()) ** 2
        weights = repeat_op_sq / (self.num_classes - 1) ** 2

        # Numerator and denominator for the kappa calculation
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.outer(hist_rater_a, hist_rater_b) / y_pred.size(0)
        denom = torch.sum(weights * expected_probs)

        return nom / (denom + self.epsilon)

    def forward(self, y_pred, y_true):
        """
        Forward pass for loss calculation.
        """
        return self.kappa_loss(y_pred, y_true)

# def make_cost_matrix(num_ratings):
#     """
#     Create a quadratic cost matrix of num_ratings x num_ratings elements.

#     :param num_ratings: number of ratings (classes).
#     :return: cost matrix.
#     """
#     cost_matrix = torch.arange(num_ratings).repeat(num_ratings, 1)
#     cost_matrix = (cost_matrix - cost_matrix.T).float() ** 2 / (num_ratings - 1) ** 2.0
#     return cost_matrix


# def qwk_loss_base(cost_matrix):
#     """
#     Compute QWK loss function.
#     :param cost_matrix: cost matrix.
#     :return: QWK loss function.
#     """
#     def _qwk_loss_base(true_prob, pred_prob):
#         targets = torch.argmax(true_prob, dim=1)
#         costs = cost_matrix[targets]

#         numerator = costs * pred_prob
#         numerator = torch.sum(numerator)

#         sum_prob = torch.sum(pred_prob, dim=0)
#         n = torch.sum(true_prob, dim=0)

#         a = torch.matmul(cost_matrix, sum_prob.view(-1, 1)).view(-1)
#         b = n / torch.sum(n)

#         epsilon = 1e-9

#         denominator = a * b
#         denominator = torch.sum(denominator) + epsilon
        
#         loss = numerator / denominator

#         return loss * 1000
    
#     return _qwk_loss_base

# def qwk_loss(cost_matrix, num_classes):
#     """
#     Compute QWK loss function.

#     :param cost_matrix: cost matrix.
#     :param num_classes: number of classes.
#     :return: QWK loss value.
#     """
#     def _qwk_loss(true_labels, pred_prob):
#         # Convert true_labels to one-hot encoding
#         true_prob = torch.nn.functional.one_hot(true_labels, num_classes).float()
        
#         # Compute the costs for the targets
#         targets = torch.argmax(true_prob, dim=1)
#         costs = cost_matrix[targets]

#         # Compute the numerator
#         numerator = costs * pred_prob
#         numerator = torch.sum(numerator)

#         # Compute sum_prob and n
#         sum_prob = torch.sum(pred_prob, dim=0)
#         n = torch.sum(true_prob, dim=0)

#         # Compute the denominator
#         a = torch.matmul(cost_matrix, sum_prob.view(-1, 1)).view(-1)
#         b = n / torch.sum(n)

#         epsilon = 1e-9

#         denominator = a * b
#         denominator = torch.sum(denominator) + epsilon

#         return numerator / denominator

#     return _qwk_loss