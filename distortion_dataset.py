import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from distortion import distort, undistort


def _convert_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

def _convert_to_opencv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


class DistortionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        wnids = os.listdir(self.root_dir)
        for wnid in wnids:
            image_path_wnid = glob.glob(os.path.join(self.root_dir, wnid, "*.jpg"))
            self.image_paths.extend(image_path_wnid)


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        # load image file as PIL image
        img_file_name = self.image_paths[idx]
        with open(img_file_name, "rb") as img_file:
            image = Image.open(img_file)
            image = image.convert('RGB')

        # make image square and crop out center region
        image = transforms.Resize(512)(image)
        image = transforms.CenterCrop(512)(image)

        image = _convert_to_opencv(image)

        # apply random radial distortion
        #distortion_coefficient = np.random.randint(0, 101)  # uniform [0, 100]
        distortion_coefficient = 100
        image_distorted = distort(image, -4e-8*distortion_coefficient, dx=0, dy=0)
        distortion_coefficient = torch.tensor(distortion_coefficient, dtype=torch.int64)

        # extract edges via canny edge detector
        image_edges = cv2.Canny(image_distorted, 50, 200, None, 3)

        image_distorted = _convert_to_pil(image_distorted)
        image_edges = _convert_to_pil(image_edges)

        # crop out smaller central region to get rid of border
        # at maximum distortion (k=-4e-6) this size ensures the border is not included
        image_distorted = transforms.CenterCrop(370)(image_distorted)
        image_edges = transforms.CenterCrop(370)(image_edges)

        # apply other transforms
        if self.transform:
            image_distorted = self.transform(image_distorted)
            image_edges = self.transform(image_edges)

        # convert image to tensor
        image_distorted = transforms.ToTensor()(image_distorted)
        image_edges = transforms.ToTensor()(image_edges)

        # normalize (needed for pretrained backbone)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_distorted = normalize(image_distorted)

        return image_distorted, image_edges, distortion_coefficient
