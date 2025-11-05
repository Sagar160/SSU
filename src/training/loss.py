import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunctions:
    def __init__(self, loss_fn_name):
        if not hasattr(self, loss_fn_name):
            raise ValueError(f"Loss function '{loss_fn_name}' is not defined.")
        self.loss_fn = getattr(self.__class__, loss_fn_name)

    @staticmethod
    def mse_loss(prediction, target):
        return F.mse_loss(prediction, target)

    @staticmethod
    def cross_entropy_loss(prediction, target):
        return F.cross_entropy(prediction, target)

    @staticmethod
    def l1_loss(prediction, target):
        return F.l1_loss(prediction, target)

    @staticmethod
    def nll_loss(prediction, target):
        return F.nll_loss(prediction, target)

    @staticmethod
    def bce_loss(prediction, target):
        return F.binary_cross_entropy(prediction, target)

    @staticmethod
    def bce_with_logits_loss(prediction, target):
        return F.binary_cross_entropy_with_logits(prediction, target)
    
    @staticmethod
    def huber_loss(prediction, target, delta=0.005):
        return F.huber_loss(prediction, target, delta=delta)
    
    @staticmethod
    def l1_sign_loss(prediction, target, sign_weight=1.0):
        # Standard L1 loss
        l1 = F.l1_loss(prediction, target)

        # Sign loss: penalize when sign is different
        sign_target = torch.sign(target)
        penalty = torch.clamp(- sign_target * prediction, min=0.0)
        sign_loss = penalty.mean()

        # Combine both losses
        return l1 + sign_weight * sign_loss
    
    @staticmethod
    def mse_sign_loss(prediction, target, sign_weight=1.0):
        # Standard L2 loss
        l2 = F.mse_loss(prediction, target)

        # Sign loss: penalize when sign is different
        sign_target = torch.sign(target)
        penalty = torch.clamp(- sign_target * prediction, min=0.0)
        sign_loss = penalty.mean()

        # Combine both losses
        return l2 + sign_weight * sign_loss
    
    @staticmethod
    def custom_loss(prediction, target, cost=None, is_val=True):
        if is_val:
            return F.mse_loss(prediction, target)
        
        # MSE Loss
        mse_loss = F.mse_loss(prediction, target)
        eps = 0.01
        
        # Vanilla RMSLE
        diff = torch.log(prediction.abs() + eps) - torch.log(target.abs() + eps)
        l_pct = torch.mean(diff**2)

        return l_pct