import time
import torch
from torch import nn, optim
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from PIL import ImageFile
from PIL import Image
import random
import os

car_file = './train'

img_transforms = transforms.Compose([transforms.RandomChoice([transforms.RandomResizedCrop((768,1024),scale=(0.8,1.0),ratio=(0.75,1.3333333),interpolation=2),
                                    transforms.RandomHorizontalFlip(p=0.2),
                                    transforms.ColorJitter(brightness=random.uniform(0.8,1.2),contrast=random.uniform(0.8,1.2),
                                                            saturation=random.uniform(0.8,1.2)),
                                    transforms.RandomGrayscale(p=0.1)
                                    ])])

img_dataset = datasets.ImageFolder(car_file,transform=img_transforms)
for idx, (img_path, img_label) in enumerate(img_dataset.imgs):
    img_name = img_path.split('/')[-1]
    img = Image.open(img_path)
    img_transformed = img_transforms(img)
    img_transformed.show()


