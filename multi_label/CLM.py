import torch
import torch.nn as nn


class CLM(nn.Module):
    def __init__(self, num_classes, link_function, min_distance=0.35, use_slope=False, fixed_thresholds=False):
        super(CLM, self).__init__()
        self.num_classes = num_classes
        self.link_function = link_function
        self.min_distance = min_distance
        self.use_slope = use_slope
        self.fixed_thresholds = fixed_thresholds
        
        #if we dont have fixed thresholds, we initalize two trainable parameters 
        if not self.fixed_thresholds:
            #first threshold
            self.thresholds_b = nn.Parameter(torch.rand(1) * 0.1) #random number between 0 and 1
            #squared distance 
            self.thresholds_a = nn.Parameter(
                torch.sqrt(torch.ones(num_classes - 2) / (num_classes - 2) / 2) * torch.rand(num_classes - 2)
            )

        if self.use_slope:
            self.slope = nn.Parameter(torch.tensor(100.0))
            
    def convert_thresholds(self, b, a, min_distance=0.35):
        a = a.pow(2) + min_distance
        thresholds_param = torch.cat([b, a], dim=0).float()
        
        band_matrix = torch.tril(torch.ones((self.num_classes - 1, self.num_classes - 1)))
        
        # Tile and reshape thresholds_param
        thresholds_param_tiled = thresholds_param.repeat(self.num_classes - 1)
        thresholds_param_reshaped = thresholds_param_tiled.view(self.num_classes - 1, self.num_classes - 1)
        
        # Element-wise multiplication
        cumulative_sums_matrix = band_matrix * thresholds_param_reshaped
        
        # Sum along the rows
        th = cumulative_sums_matrix.sum(dim=1)
        return th
    
    def nnpom(self, projected, thresholds):
        projected = projected.view(-1).float()
        
        if self.use_slope:
            projected = projected * self.slope
            thresholds = thresholds * self.slope

        m = projected.shape[0]
        a = thresholds.repeat(m, 1)
        b = projected.repeat(self.num_classes - 1, 1).t()
        z3 = a - b

        if self.link_function == 'probit':
            a3T = torch.distributions.Normal(0, 1).cdf(z3)
        elif self.link_function == 'cloglog':
            a3T = 1 - torch.exp(-torch.exp(z3))
        else:  # logit
            a3T = torch.sigmoid(z3)

        ones = torch.ones((m, 1))
        a3 = torch.cat([a3T, ones], dim=1)
        a3 = torch.cat([a3[:, :1], a3[:, 1:] - a3[:, :-1]], dim=1)

        return a3

    def forward(self, x):
        if self.fixed_thresholds:
            thresholds = torch.linspace(0, 1, self.num_classes, dtype=torch.float32)[1:-1]
        else:
            thresholds = self.convert_thresholds(self.thresholds_b, self.thresholds_a, self.min_distance)

        return self.nnpom(x, thresholds)

    def extra_repr(self):
        return 'num_classes={}, link_function={}, min_distance={}, use_slope={}, fixed_thresholds={}'.format(
            self.num_classes, self.link_function, self.min_distance, self.use_slope, self.fixed_thresholds
        )
