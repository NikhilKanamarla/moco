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

def get_digits(str1):
    c = ""
    for i in str1:
        if i.isdigit():
            c += i
    return c

def main():
    path = '/datac/nkanama/RetinaFace/save_folder_FFHQ/crops'
    listOfImagePaths = []

    # iterate through the names of contents of the folder
    for image_path in os.listdir(path):
        # create the full input path and read the file
        input_path = os.path.join(path, image_path)
        listOfImagePaths.append(input_path)

    with open('/datac/nkanama/RetinaFace/save_folder_FFHQ/text_files/masterFFHQ.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image1", "image2", "label"])
        #pdb.set_trace()
        for image1 in listOfImagePaths:
            if(image1.find("leftEye") != -1):
                image_num = get_digits(image1)
                for image2 in listOfImagePaths:
                    if(image2.find("rightEye") != -1):
                        if(image_num == get_digits(image2)):
                            writer.writerow([image1,image2,"postive"])
                        else:
                            writer.writerow([image1,image2,"negative"])



if __name__ == '__main__':
    main()
