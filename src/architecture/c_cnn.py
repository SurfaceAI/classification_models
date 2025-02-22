import sys

sys.path.append(".")

import torch
import torch.nn as nn
from torchvision import models

# from src.utils.helper import *
from src import constants as const
import copy


class C_CNN(nn.Module):
    def __init__(
        self,
        base_model,
        number_of_last_blocks,
        # head_coarse,
        head_fine,
        # num_c,
        num_f,
    ):  # TODO: args/kwargs
        super(C_CNN, self).__init__()  # TODO: args kwargs
        """
        Parameters:
            - base_model (efficientnet, resnet))
            - number_of_last_blocks (0, 1, 2..)
            # - head_coarse (classification, regression) defines last classifier layer with num_classes + criterion
            - head_fine (classification, regression) -"-
            # - num_c
            - num_f (len(num_f) = num_c)
        """

        # 1 as default num classes for feature layers
        if number_of_last_blocks == 0:
            self.common_blocks = base_model(1).features
            self.coarse_blocks = nn.Identity()
            self.fine_blocks = nn.Identity()
        else:
            self.common_blocks = base_model(1).features[:-number_of_last_blocks]
            self.coarse_blocks = base_model(1).features[-number_of_last_blocks:]
            self.fine_blocks = base_model(1).features[-number_of_last_blocks:]

        # self.head_coarse = head_coarse
        self.head_fine = head_fine

        self.num_c = len(num_f)
        self.num_f = num_f

        # coarse
        self.coarse_classifier = base_model(self.num_c).classifier
        self.coarse_criterion = nn.CrossEntropyLoss

        # fine
        if self.head_fine == const.HEAD_CLASSIFICATION:
            self.fine_classifier = nn.ModuleList([base_model(num).classifier for num in self.num_f])
            self.fine_criterion = nn.CrossEntropyLoss
        elif self.head_fine == const.HEAD_REGRESSION:
            self.fine_classifier = nn.ModuleList([base_model(1).classifier for _ in self.num_f])
            self.fine_criterion = nn.MSELoss
        else:
            raise ValueError(f"Fine head {self.head_fine} not applicable!")

    def get_prediction_values(self, x_coarse=None, x_fine=None):
        if x_coarse is not None:
            x_coarse = nn.functional.softmax(x_coarse, dim=1)

        if x_fine is not None:
            if self.head_fine == const.HEAD_CLASSIFICATION:
                x_fine = nn.functional.softmax(x_fine, dim=1)
            elif self.head_fine == const.HEAD_REGRESSION:
                x_fine = x_fine
            else:
                raise ValueError(f"Fine head {self.head_fine} not applicable!")
        return x_coarse, x_fine

    def get_prediction_indices(self, x_coarse=None, x_fine=None):
        if x_coarse is not None:
            x_coarse = nn.functional.softmax(x_coarse, dim=1)
            x_coarse = torch.argmax(x_coarse, dim=1)

        if x_fine is not None:
            if self.head_fine == const.HEAD_CLASSIFICATION:
                x_fine = nn.functional.softmax(x_fine, dim=1)
                x_fine = torch.argmax(x_fine, dim=1)
            elif self.head_fine == const.HEAD_REGRESSION:
                x_fine = x_fine
                x_fine = x_fine.round().to(torch.int64)
            else:
                raise ValueError(f"Fine head {self.head_fine} not applicable!")
        return x_coarse, x_fine

    def forward(self, x, gt_coarse=None):
        x = self.common_blocks(x)

        x_coarse = self.coarse_blocks(x)
        x_coarse = nn.AdaptiveAvgPool2d(1)(x_coarse)
        x_coarse = torch.flatten(x_coarse, 1)
        x_coarse_output = self.coarse_classifier(x_coarse)

        x_fine = self.fine_blocks(x)
        x_fine = nn.AdaptiveAvgPool2d(1)(x_fine)
        x_fine = torch.flatten(x_fine, 1)
        if gt_coarse is None:
            gt_coarse = torch.argmax(
                self.get_prediction_indices(x_coarse=x_coarse_output)[0], dim=1
            )

        # gt_coarse = gt_coarse.unsqueeze(1)
        # x_fine = torch.cat([self.fine_classifier[i](x_fine.clone()) for i in range(self.num_c)], dim=1)
        # x_fine = torch.gather(x_fine, dim=1, index=gt_coarse)
        # alternative
        selected_classifiers = [self.fine_classifier[i] for i in gt_coarse.tolist()]
        x_fine_output = torch.stack(
            [clf(x.clone()) for clf, x in zip(selected_classifiers, x_fine)]
        )

        if self.head_fine == const.HEAD_REGRESSION:
            x_fine_output = x_fine_output.flatten()

        return x_coarse_output, x_fine_output

    def get_optimizer_layers(self):
        return self.coarse_classifier, *self.fine_classifier

    # def get_prediction_values(self, x_coarse=None, x_fine=None):
    #     if x_coarse is not None:
    #         x_coarse_value = nn.functional.softmax(x_coarse, dim=1)

    #     if x_fine is not None:
    #         if self.head_fine == const.HEAD_CLASSIFICATION:
    #             x_fine_value = nn.functional.softmax(x_fine, dim=1)
    #         elif self.head_fine == const.HEAD_REGRESSION:
    #             x_fine_value = x_fine
    #         else:
    #             raise ValueError(f"Fine head {self.head_fine} not applicable!")
    #     return x_coarse_value, x_fine_value

    # def get_prediction_indices(self, x_coarse=None, x_fine=None):
    #     if x_coarse is not None:
    #         x_coarse_value = nn.functional.softmax(x_coarse, dim=1)
    #         x_coarse_value = torch.argmax(x_coarse_value, dim=1)

    #     if x_fine is not None:
    #         if self.head_fine == const.HEAD_CLASSIFICATION:
    #             x_fine_value = nn.functional.softmax(x_fine, dim=1)
    #             x_fine_value = torch.argmax(x_fine_value, dim=1)
    #         elif self.head_fine == const.HEAD_REGRESSION:
    #             x_fine_value = x_fine
    #             x_fine = x_fine_value.round().to(torch.int64)
    #         else:
    #             raise ValueError(f"Fine head {self.head_fine} not applicable!")
    #     return x_coarse_value, x_fine_value

    # def forward(self, x, gt_coarse=None):
    #     x = self.common_blocks(x)

    #     x_coarse = self.coarse_blocks(x)
    #     x_coarse = nn.AdaptiveAvgPool2d(1)(x_coarse)
    #     # x_coarse = torch.flatten(x_coarse, 1)
    #     x_coarse_flat = x_coarse.reshape(x_coarse.size(0), -1)
    #     x_coarse_pred = self.coarse_classifier(x_coarse_flat)

    #     x_fine = self.fine_blocks(x)
    #     x_fine = nn.AdaptiveAvgPool2d(1)(x_fine)
    #     # x_fine = torch.flatten(x_fine, 1)
    #     x_fine_flat = x_fine.reshape(x_fine.size(0), -1)
    #     x_fine_pred = torch.cat([self.fine_classifier[i](x_fine_flat.clone()) for i in range(self.num_c)], dim=1)
    #     if gt_coarse is None:
    #         gt_coarse = torch.argmax(self.get_prediction_indices(x_coarse=x_coarse_pred)[0], dim=1)
    #     gt_coarse = gt_coarse.unsqueeze(1)
    #     x_fine_pred = torch.gather(x_fine_pred, dim=1, index=gt_coarse)
    #     if self.head_fine == const.HEAD_REGRESSION:
    #         x_fine_pred = x_fine_pred.flatten()

    #     return x_coarse_pred, x_fine_pred
