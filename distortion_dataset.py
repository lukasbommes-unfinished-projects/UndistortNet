import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch

def _distort_radial(image, distortion_coefficient):

    return image_distorted


class DistortionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = glob.glob(os.path.join(self.root_dir, "*.JPEG"))
        print(self.image_names)


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        # load image file from disk
        img_file_name = self.image_names[idx]
        #with open(img_file_name, "rb") as img_file:
        #    image = Image.open(img_file)
        #    image = image.convert('RGB')

        # open as opencv image
        image = cv2.imread(img_file_name)

        # apply random radial distortion
        distortion_coefficient = np.random.rand()
        image = _distort_radial(image, distortion_coefficient)

        # convert to PIl image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # apply transforms
        if self.transform:
            image = self.transform(image)

        distortion_coefficient = 0.0

        return image, distortion_coefficient
