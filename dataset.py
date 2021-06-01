import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# This would probably need modification for another dataset
mean = np.array ([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    for c in range(3):
        tensors[:,c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

class ImageDataset(Dataset):
    def __init__(self,root,hrShape):
        hrHeight, hrWidth = hrShape
        self.lrTransform = transforms.Compose([
            transforms.Resize((hrHeight // 4, hrHeight // 4),Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.hrTransform = transforms.Compose([
            transforms.Resize((hrHeight, hrHeight), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        imgLr = self.lrTransform(img)
        imgHr = self.hrTransform(img)

        return{"lr": imgLr, "hr": imgHr}

    def __len__(self):
        return len(self.files)