import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from distortion import distort, undistort


class DistortionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = glob.glob(os.path.join(self.root_dir, "*.JPEG"))


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        # load image file as PIL image
        img_file_name = self.image_names[idx]
        with open(img_file_name, "rb") as img_file:
            image = Image.open(img_file)
            image = image.convert('RGB')

        # make image square and crop out center region
        pre_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
        image = pre_transform(image)

        # open as opencv image
        #image = cv2.imread(img_file_name)

        # convert to openCV image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # apply random radial distortion
        distortion_coefficient = np.random.randint(0, 101)  # uniform [0, 100]
        image = distort(image, -10e-8*distortion_coefficient, dx=0, dy=0)

        # convert to PIL image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # apply transforms
        if self.transform:
            image = self.transform(image)

        return image, distortion_coefficient
