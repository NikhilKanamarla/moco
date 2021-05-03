import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

from PIL import ImageFilter
import random
from torch.utils.data.dataset import Dataset
import pandas as pd
import csv
import pdb
from pathlib import Path

parser = argparse.ArgumentParser(description='Put crops into CSV')
parser.add_argument('--path', type=str, required=True, help='path to image crops')
parser.add_argument('--train', type=str, required=True, help='path to new train csv')
parser.add_argument('--val', type=str, required=True, help='path to new val csv')
parser.add_argument('--pathF', type=str, required=True, help='path to image crops')
parser.add_argument('--test', type=str, required=True, help='path to new test csv')
args = parser.parse_args()


def main():
    path = args.path
    pathF = args.pathF
    gpuTrain = args.train
    gpuVal = args.val
    gpuTest = args.test

    listOfTrainImagePaths = []
    listofValImagePaths = []
    listofTestImagePaths = []
    # iterate through the names of contents of the folder
    contentsOfFolder = sorted(Path(path).iterdir(), key=os.path.getmtime)
    #create train set 42,000 images (4 crops per image)
    for image_path in range(0,int(len(contentsOfFolder)*0.6)):
        # create the full input path and read the file
        input_path = os.path.join(path, contentsOfFolder[image_path])
        listOfTrainImagePaths.append(input_path)
    #create validation set 10,000 images (4 crops per image)
    for image_path in range(int(len(contentsOfFolder)*0.6), int((len(contentsOfFolder)*0.6) + 40000)):
        # create the full input path and read the file
        input_path = os.path.join(path, contentsOfFolder[image_path])
        listofValImagePaths.append(input_path)
    #create test csv
    for image_path in range(int(len(contentsOfFolder)*0.6)+40000, len(contentsOfFolder)):
        # create the full input path and read the file
        input_path = os.path.join(path, contentsOfFolder[image_path])
        listofTestImagePaths.append(input_path)
    contentsOfFakeFolder = sorted(Path(args.pathF).iterdir(), key=os.path.getmtime)
    for image_path in range(0, 70000):
        # create the full input path and read the file
        input_path = os.path.join(args.pathF, contentsOfFakeFolder[image_path])
        listofTestImagePaths.append(input_path)
    
    #create test set 
    #load into hashmap for quick access
    hashTrainImages = set(listOfTrainImagePaths)
    hashValImages = set(listofValImagePaths)
    hashTestImages = set(listofTestImagePaths)

    #write to CSV for train file
    with open(gpuTrain, 'w', newline='') as file:
        writer = csv.writer(file)
        #only store postive samples
        writer.writerow(["image1", "image2"])
        for image1 in listOfTrainImagePaths:
            
            #image pairings between left and right eye
            if(image1.find("leftEye") != -1):
                image2 = image1.replace("leftEye","rightEye")
                if(image2 in hashTrainImages):
                    writer.writerow([image1,image2])
             

    #write to CSV for validation file
    with open(gpuVal, 'w', newline='') as file:
        writer = csv.writer(file)
        #only store postive samples
        writer.writerow(["image1", "image2"])
        for image1 in listofValImagePaths:
            #image pairings between left and right eye
            if(image1.find("leftEye") != -1):
                image2 = image1.replace("leftEye","rightEye")
                if(image2 in hashValImages):
                    writer.writerow([image1,image2])
    
    #write to CSV for test file
    with open(gpuTest, 'w', newline='') as file:
        writer = csv.writer(file)
        #only store postive samples
        writer.writerow(["image1", "image2"])
        for image1 in listofTestImagePaths:
            #image pairings between left and right eye
            if(image1.find("leftEye") != -1):
                image2 = image1.replace("leftEye","rightEye")
                if(image2 in hashTestImages):
                    writer.writerow([image1,image2])


if __name__ == '__main__':
    main()
