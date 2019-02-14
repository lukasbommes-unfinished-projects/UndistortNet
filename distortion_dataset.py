import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from distortion import compute_maps, distort_image, undistort_image, crop_max


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
        # distortion parameter ranges
        self.ks = np.array([-4*x*1e-3 for x in range(0, 101, 5)])
        self.dxs = np.array(range(-50, 51, 5))
        self.dys = np.array(range(-50, 51, 5))
        # image size (squre)
        self.image_size = 512


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

        # sample randomly from possible distortion parameters
        k = np.random.choice(self.ks)
        dx = np.random.choice(self.dxs)
        dy = np.random.choice(self.dys)

        maps = compute_maps(self.image_size, self.image_size, k, dx, dy)
        image_distorted = distort_image(image, maps)
        image_undistorted = undistort_image(image_distorted, maps)
        k = torch.tensor(k, dtype=torch.int64)
        dx = torch.tensor(k, dtype=torch.int64)
        dy = torch.tensor(k, dtype=torch.int64)

        image_distorted_cropped = crop_max()

        image_distorted = _convert_to_pil(image_distorted)
        image_undistorted = _convert_to_pil(image_undistorted)

        # crop out smaller central region to get rid of border
        # at maximum distortion (k=-4e-6) this size ensures the border is not included
        image_distorted = transforms.CenterCrop(370)(image_distorted)

        # apply other transforms
        if self.transform:
            image_distorted = self.transform(image_distorted)

        # convert image to tensor
        image_distorted = transforms.ToTensor()(image_distorted)

        # normalize (needed for pretrained backbone)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_distorted = normalize(image_distorted)

        return image_distorted, distortion_coefficient
