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

def main():
    path = '/datac/nkanama/RetinaFace/save_folder_FFHQ/crops'
    listOfTrainImagePaths = []
    listOfValImagePaths = []
    # iterate through the names of contents of the folder
    contentsOfFolder = sorted(Path(path).iterdir(), key=os.path.getmtime)

    #create train set 35,000 images (4 crops per image)
    #pdb.set_trace()
    for image_path in range(0,int(len(contentsOfFolder)/2)):
        # create the full input path and read the file
        input_path = os.path.join(path, contentsOfFolder[image_path])
        listOfTrainImagePaths.append(input_path)
    #create validation set 10,000 images (4 crops per image)
    for image_path in range(int(len(contentsOfFolder)/2), int(len(contentsOfFolder)/2) + 40000):
        # create the full input path and read the file
        input_path = os.path.join(path, contentsOfFolder[image_path])
        listOfValImagePaths.append(input_path)

    #write to CSV for train file
    with open('/datac/nkanama/RetinaFace/save_folder_FFHQ/text_files/trainFFHQ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        #only store postive samples
        writer.writerow(["image1", "image2"])
        for image1 in listOfTrainImagePaths:
            if(image1.find("leftEye") != -1):
                image2 = image1.replace("leftEye","rightEye")
                writer.writerow([image1,image2])

    #write to CSV for validation file
    with open('/datac/nkanama/RetinaFace/save_folder_FFHQ/text_files/valFFHQ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        #only store postive samples
        writer.writerow(["image1", "image2"])
        for image1 in listOfValImagePaths:
            if(image1.find("leftEye") != -1):
                image2 = image1.replace("leftEye","rightEye")
                writer.writerow([image1,image2])
                



if __name__ == '__main__':
    main()
