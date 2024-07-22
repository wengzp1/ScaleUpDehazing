import random
import numpy as np
import torch.nn as nn
from torchvision import transforms as T
class AugNoneOpt(nn.Module):
    def __init__(self):
        super(AugNoneOpt, self).__init__()
        self.weak_aug = nn.Sequential(T.CenterCrop(16))
        self.aggr_aug = nn.Sequential(T.GaussianBlur(kernel_size=(9,21),sigma=(0.1,5)))
        

    def forward(self, source_img):
        augweak_sourge_img =self.weak_aug(source_img)
        augaggr_sourge_img =self.aggr_aug(augweak_sourge_img)


        return augweak_sourge_img, augaggr_sourge_img
