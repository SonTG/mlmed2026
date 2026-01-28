import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds = (preds > 0.5).float()
        targets = (targets > 0.5).float()

        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return bce_loss + dice_loss
