import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFilter
import random
from torch.utils.data.dataset import Dataset
import pandas as pd
import csv
import pdb
import numpy as np
from PIL import Image
from skimage import io, transform
#import loader
#import builder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=1)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column contains the second image
        self.image_second_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        #resize
        self.transform = transform

    def __getitem__(self, index):
        # Get image name from the pandas df
        first_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(first_image_name)
        # Get image name from the pandas df
        second_image_name = self.image_second_arr[index]
        # Open image
        img_as_img_second = Image.open(second_image_name)

        # If there is an operation
        #pdb.set_trace()
        if self.transform:
            img_as_img = self.transform(img_as_img)
            img_as_img_second = self.transform(img_as_img_second)

        sample = (img_as_img, img_as_img_second)

        return sample

    def __len__(self):
        return self.data_len
    



def main(argument):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
            transforms.Resize((256,256)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    transformations = transforms.Compose(augmentation)
    ffhq_dataset = CustomDatasetFromImages(argument, transformations)

    for i in range(len(ffhq_dataset)):
        sample = ffhq_dataset[i]

        print(i, sample[0].shape, sample[1].shape)

        if i == 10:
            #plt.savefig(i)
            break

def getLoader(argument, transform):
    ffhq_dataset = CustomDatasetFromImages(argument, transform)
    return ffhq_dataset
        

if __name__ == '__main__':
    arg = '/datac/nkanama/RetinaFace/save_folder_FFHQ/text_files/trainFFHQ.csv' 
    main(arg)
