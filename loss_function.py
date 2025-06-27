"""
以二分类任务为例
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss for binary classification.
        
        Args:
            alpha (float): Class balance factor. Default: 0.25
            gamma (float): Focusing parameter. Default: 2
            reduction (str): Reduction method. Options: 'none', 'mean', 'sum'. Default: 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Focal Loss.
        """
        # 计算二元交叉熵损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算 p_t
        pt = torch.exp(-BCE_loss)
        
        # 计算 Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # 应用 reduction 方法
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
