import random
import numpy as np
import torch.nn as nn
from torchvision import transforms as T
class AugNoneOpt(nn.Module):
    def __init__(self):
        super(AugNoneOpt, self).__init__()
        self.aggr_aug = nn.Sequential(T.ColorJitter(brightness=0.5, hue=0.3))
        self.weak_aug = nn.Sequential(T.RandomCrop(128),
                         T.RandomHorizontalFlip(p=0.5))

    def forward(self, source_img):
        augweak_sourge_img =self.weak_aug(source_img)
        augaggr_sourge_img =self.aggr_aug(augweak_sourge_img)


        return augweak_sourge_img, augaggr_sourge_img