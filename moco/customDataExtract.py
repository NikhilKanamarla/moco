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
        #only store postive samples
        writer.writerow(["image1", "image2"])
        #pdb.set_trace()
        for image1 in listOfImagePaths:
            if(image1.find("leftEye") != -1):
                image2 = image1.replace("leftEye","rightEye")
                writer.writerow([image1,image2])
                



if __name__ == '__main__':
    main()
