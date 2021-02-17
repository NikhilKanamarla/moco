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
        # Third column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
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
        if self.transform:
            img_as_img = self.transform(img_as_img)
            img_as_img_second = self.transform(img_as_img_second)

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        img_as_tensor_second = self.to_tensor(img_as_img_second)

        # Get label(class) of the image based on the cropped pandas column
        image_pair_label = self.label_arr[index]

        sample = (img_as_tensor, img_as_tensor_second, image_pair_label)

        return sample

    def __len__(self):
        return self.data_len
    



def main(argument):
    transformations = transforms.Compose([transforms.Resize((256,256))])
    ffhq_dataset = CustomDatasetFromImages(argument, transformations)

    for i in range(len(ffhq_dataset)):
        sample = ffhq_dataset[i]

        print(i, sample[0].shape, sample[1].shape, sample[2])

        '''
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        '''
        if i == 10:
            #plt.savefig(i)
            break

def getLoader(argument, transform):
    ffhq_dataset = CustomDatasetFromImages(argument, transform)
    return ffhq_dataset
        

if __name__ == '__main__':
    arg = '/datac/nkanama/RetinaFace/save_folder_FFHQ/text_files/masterFFHQ.csv' 
    main(arg)
