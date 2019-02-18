import numpy as np
import cv2
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

from distortion_dataset import DistortionDataset


class UndistortLayer(nn.Module):
    def __init__(self):
        super(UndistortLayer, self).__init__()

    def forward(self, im_d, k, dx, dy):
        """This function takes the distorted input image, a set of distortion
        parameters and outputs an undistorted image.

        Args:
            im_d (torch.Tensor): Batch of distorted input images of shape [B X C X H X W] and
                dtype torch.uint9.
            k (torch.Tensor): Distortion coefficients of shape [B x 1]. Should be
                in range [-0.4, 0). Expected data type is torch.float32.
            dx (torch.Tensor): Distortion center offsets in x direction. Should have
                shape [B x 1] and lie in range [-50, 50]. Expected data type is torch.float32.
            dy (torch.Tensor): Distortion center offsets in x direction. Should have
                shape [B x 1] and lie in range [-50, 50]. Expected data type is torch.float32.

        Returns:
            im_ud (torch.Tensor): Batch of undistorted output images with shape [B X C X H X W]
            and dtype torch.uint8.
        """
        im_ud = torch.zeros(im_d.shape, dtype=im_d.dtype)
        # compute xd, yd for each xu, yu and each channel as well as element in the batch
        b, c, h, w = im_d.shape
        bv, cv, yu, xu = torch.meshgrid(torch.arange(0, b, dtype=torch.long),
                                        torch.arange(0, c, dtype=torch.long),
                                        torch.arange(0, h, dtype=torch.long),
                                        torch.arange(0, w, dtype=torch.long))
        xu = xu.type(torch.float)
        yu = yu.type(torch.float)

        # bring parameter tensors in shape [b, c, h, w]
        k = k.squeeze().expand([c, h, w, -1]).permute(3, 0, 1, 2)
        dx = dx.squeeze().expand([c, h, w, -1]).permute(3, 0, 1, 2)
        dy = dy.squeeze().expand([c, h, w, -1]).permute(3, 0, 1, 2)

        # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
        #print(dx.shape)
        xur = (xu - dx)/w - 1/2
        yur = (yu - dy)/h - 1/2
        # convert to polar
        ru = torch.sqrt(xur * xur + yur * yur)
        theta = torch.atan2(yur, xur)
        # distort coordinates
        rd = ru / (1 - k * ru * ru)
        # convert back to cartesian
        xdr = rd * torch.cos(theta)
        ydr = rd * torch.sin(theta)
        # un-normalize coordinates to oriaginal range [0..w x 0...h]
        xd = (xdr + 1/2)*w + dx
        yd = (ydr + 1/2)*h + dy

        # bilinear interpolation of im_ud(xu, yu) based on pixel values in im_d
        omega_x = xd - torch.floor(xd)
        omega_y = yd - torch.floor(yd)
        omega_nx = 1 - omega_x
        omega_ny = 1 - omega_y

        # convert to int so can be used for indexing
        xd_floor = torch.floor(xd).type(torch.int64)
        xd_ceil = torch.ceil(xd).type(torch.int64)
        yd_floor = torch.floor(yd).type(torch.int64)
        yd_ceil = torch.ceil(yd).type(torch.int64)

        # get intensity values from distorted image
        im_d_0 = im_d[bv, cv, yd_floor, xd_floor]
        im_d_1 = im_d[bv, cv, yd_floor, xd_ceil]
        im_d_2 = im_d[bv, cv, yd_ceil, xd_floor]
        im_d_3 = im_d[bv, cv, yd_ceil, xd_ceil]

        # compute new intensity values for output image
        xu_idx = xu.type(torch.int64)
        yu_idx = yu.type(torch.int64)
        im_ud[bv, cv, yu_idx, xu_idx] = (omega_nx*omega_ny*im_d_0 +
                                 omega_x*omega_ny*im_d_1 +
                                 omega_nx*omega_y*im_d_2 +
                                 omega_x*omega_y*im_d_3)

        return im_ud#im_ud.type(torch.uint8)


if __name__ == "__main__":


    undistort_layer = UndistortLayer()
    input_img = cv2.imread("distorted_img_k=-0.4_dx=50_dy=-10.jpg")
    h, w, c = input_img.shape

    k = torch.tensor([[-0.4], [-0.0001]], dtype=torch.float)
    dx = torch.tensor([[50], [50]], dtype=torch.float)
    dy = torch.tensor([[-10], [-10]], dtype=torch.float)

    # convert input image to tensor
    input_img_m = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img_m = PIL.Image.fromarray(input_img_m)
    input_tensor = transforms.ToTensor()(input_img_m)
    input_tensor = input_tensor.unsqueeze(0).expand(2, c, h, w)
    output = undistort_layer(input_tensor, k, dx, dy)

    output = np.array(output)
    output1 = output[0, :, :, :]
    output1 = np.moveaxis(output1, 0, 2)
    output_img1 = cv2.cvtColor(output1, cv2.COLOR_RGB2BGR)
    output2 = output[1, :, :, :]
    output2 = np.moveaxis(output2, 0, 2)
    output_img2 = cv2.cvtColor(output2, cv2.COLOR_RGB2BGR)


    while True:
        cv2.imshow("input", input_img)
        cv2.imshow("ouput1", output_img1)
        cv2.imshow("ouput2", output_img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
