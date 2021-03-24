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
    listofValImagePaths = []
    # iterate through the names of contents of the folder
    contentsOfFolder = sorted(Path(path).iterdir(), key=os.path.getmtime)

    #create train set 35,000 images (4 crops per image)
    for image_path in range(0,int(len(contentsOfFolder)*0.6)):
        # create the full input path and read the file
        input_path = os.path.join(path, contentsOfFolder[image_path])
        listOfTrainImagePaths.append(input_path)
    #create validation set 10,000 images (4 crops per image)
    for image_path in range(int(len(contentsOfFolder)*0.6), int((len(contentsOfFolder)*0.6) + 40000)):
        # create the full input path and read the file
        input_path = os.path.join(path, contentsOfFolder[image_path])
        listofValImagePaths.append(input_path)

    #write to CSV for train file
    with open('/datac/nkanama/RetinaFace/save_folder_FFHQ/text_files/trainFFHQ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        #only store postive samples
        writer.writerow(["image1", "image2"])
        #pdb.set_trace()
        for image1 in listOfTrainImagePaths:
            #pairings between left eye and other face features
            if(image1.find("leftEye") != -1):
                image2 = image1.replace("leftEye","rightEye")
                if(image2 in listOfTrainImagePaths):
                    writer.writerow([image1,image2])
                image3 = image1.replace("leftEye","mouth")
                if(image3 in listOfTrainImagePaths):
                    writer.writerow([image1,image3])
                image4 = image1.replace("leftEye","nose")
                if(image4 in listOfTrainImagePaths):
                    writer.writerow([image1,image4])
            #pairings between right eye and other face features (not including left eye)
            if(image1.find("rightEye") != -1):
                image3 = image1.replace("rightEye","mouth")
                if(image3 in listOfTrainImagePaths):
                    writer.writerow([image1,image3])
                image4 = image1.replace("rightEye","nose")
                if(image4 in listOfTrainImagePaths):
                    writer.writerow([image1,image4])
            #pairings between mouth other face features (not including left or right eye)
            if(image1.find("mouth") != -1):
                image2 = image1.replace("mouth","nose")
                if(image2 in listOfTrainImagePaths):
                    writer.writerow([image1,image2])
            
            

    #write to CSV for validation file
    with open('/datac/nkanama/RetinaFace/save_folder_FFHQ/text_files/valFFHQ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        #only store postive samples
        writer.writerow(["image1", "image2"])
        for image1 in listofValImagePaths:
            #pairings between left eye and other face features
            if(image1.find("leftEye") != -1):
                image2 = image1.replace("leftEye","rightEye")
                if(image2 in listofValImagePaths):
                    writer.writerow([image1,image2])
                image3 = image1.replace("leftEye","mouth")
                if(image3 in listofValImagePaths):
                    writer.writerow([image1,image3])
                image4 = image1.replace("leftEye","nose")
                if(image4 in listofValImagePaths):
                    writer.writerow([image1,image4])
            #pairings between right eye and other face features (not including left eye)
            if(image1.find("rightEye") != -1):
                image3 = image1.replace("rightEye","mouth")
                if(image3 in listofValImagePaths):
                    writer.writerow([image1,image3])
                image4 = image1.replace("rightEye","nose")
                if(image4 in listofValImagePaths):
                    writer.writerow([image1,image4])
            #pairings between mouth other face features (not including left or right eye)
            if(image1.find("mouth") != -1):
                image2 = image1.replace("mouth","nose")
                if(image2 in listofValImagePaths):
                    writer.writerow([image1,image2])
                



if __name__ == '__main__':
    main()
