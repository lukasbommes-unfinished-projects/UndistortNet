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
    image = np.array(image)
    # if channel first, convert to channels last
    if np.shape(image)[0] == 3:
        image = np.moveaxis(image, -1, 0)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


# def classes_to_parameters(ks_c, dxs_c, dys_c):
#     """Transform class indices to distortion parameter values.
#
#     Args:
#         ks_c (ints): class index for k, e.g. {0, 1, ..., 21}
#         dxs_c (ints): class index for dx, e.g. {0, 1, ..., 21}
#         dys_c (ints): class index for for dy, e.g. {0, 1, ..., 21}
#
#     Returns:
#         ks (float): Actual value for k, e.g. [0, -0.02, ..., -0.4]
#         dx (float): Actual value for dx, e.g. [-50, -45, ..., 50]
#         dy (float):Actual value for dy, e.g. [-50, -45, ..., 50]
#     """
#     ks = -20e-3*ks_c
#     dxs = 5*dxs_c-50
#     dys = 5*dys_c-50
#     return ks, dxs, dys
#
#
# def parameters_to_classes(ks, dxs, dys):
#     """Transform class indices to distortion parameter values.
#
#     Args:
#         ks (float): Actual value for k, e.g. [0, -0.02, ..., -0.4]
#         dx (float): Actual value for dx, e.g. [-50, -45, ..., 50]
#         dy (float):Actual value for dy, e.g. [-50, -45, ..., 50]
#
#     Returns:
#         ks_c (ints): class index for k, e.g. {0, 1, ..., 21}
#         dxs_c (ints): class index for dx, e.g. {0, 1, ..., 21}
#         dys_c (ints): class index for for dy, e.g. {0, 1, ..., 21}
#     """
#     ks_c = (ks/-20e-3).astype(np.int)
#     dxs_c = ((dxs+50)/5).astype(np.int)
#     dys_c = ((dys+50)/5).astype(np.int)
#     return ks_c, dxs_c, dys_c



class DistortionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        wnids = os.listdir(self.root_dir)
        for wnid in wnids:
            image_path_wnid = glob.glob(os.path.join(self.root_dir, wnid, "*.jpg"))
            self.image_paths.extend(image_path_wnid)
        # possible distortion parameters
        #self.ks = np.linspace(-0.4, -0.001, 100)
        #self.dxs = np.linspace(-50, 50, 100)
        #self.dys = np.linspace(-50, 50, 100)
        # image size (square)
        self.image_size = 256


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        # load image file as PIL image
        img_file_name = self.image_paths[idx]
        with open(img_file_name, "rb") as img_file:
            image = Image.open(img_file)
            image = image.convert('RGB')

        # make image square and crop out center region
        image = transforms.Resize(self.image_size)(image)
        image = transforms.CenterCrop(self.image_size)(image)

        image = _convert_to_opencv(image)

        # sample randomly from possible distortion parameters
        k = -0.4 * np.random.rand() - 0.01  # [-0.4 .. -0.01]
        dx = 0 #100 * np.random.rand() - 50  # [-50 .. 50]
        dy = 0 #100 * np.random.rand() - 50  # [-50 .. 50]

        # distort image with sampled distortion parameters
        maps = compute_maps(self.image_size, self.image_size, k, dx, dy)
        image_distorted = distort_image(image, maps)
        image_undistorted = undistort_image(image_distorted, maps)

        # crop out the maximal central region to get rid of black border
        image_distorted_cropped, coords = crop_max(image_distorted, self.image_size,
            self.image_size, maps, dx, dy)

        # convert to pyTorch compatible formats
        k_c = torch.tensor(k, dtype=torch.float32)
        dx_c = torch.tensor(dx, dtype=torch.float32)
        dy_c = torch.tensor(dy, dtype=torch.float32)
        image_distorted = _convert_to_pil(image_distorted)
        image_undistorted = _convert_to_pil(image_undistorted)
        image_distorted_cropped = _convert_to_pil(image_distorted_cropped)

        # resize the cropped distorted image back to original size
        image_distorted_cropped = transforms.Resize((self.image_size, self.image_size))(image_distorted_cropped)

        # convert image to tensor
        image_distorted = transforms.ToTensor()(image_distorted)
        image_undistorted = transforms.ToTensor()(image_undistorted)
        image_distorted_cropped = transforms.ToTensor()(image_distorted_cropped)

        # normalize (needed for pretrained backbone)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_distorted = normalize(image_distorted)
        image_undistorted = normalize(image_undistorted)
        image_distorted_cropped = normalize(image_distorted_cropped)

        data = (image_distorted, image_distorted_cropped, image_undistorted, k_c, dx_c, dy_c)

        return data
