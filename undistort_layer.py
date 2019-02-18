import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

from distortion_dataset import DistortionDataset, classes_to_parameters


class UndistortLayer(nn.Module):
    def __init__(self):
        super(UndistortLayer, self).__init__()

    def forward(self, im_d, k, dx, dy):
        """This function takes the distorted input image, a set of distortion
        parameters and outputs an undistorted image."""

        im_ud = torch.zeros(im_d.shape, dtype=im_d.dtype)
        print(im_ud.shape)

        # compute xd, yd for each xu, yu
        c, h, w = im_d.shape
        print(c, h, w)
        for xu in torch.arange(0, w, dtype=torch.float):
            for yu in torch.arange(0, h, dtype=torch.float):
                # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
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
                # check border violation
                if xd_floor >= 0 and xd_ceil < w and yd_floor >= 0 and yd_ceil < h:
                    im_d_0 = im_d[:, yd_floor, xd_floor]
                    im_d_1 = im_d[:, yd_floor, xd_ceil]
                    im_d_2 = im_d[:, yd_ceil, xd_floor]
                    im_d_3 = im_d[:, yd_ceil, xd_ceil]

                    xu_idx = xu.type(torch.int64)
                    yu_idx = yu.type(torch.int64)
                    im_ud[:, yu_idx, xu_idx] = (omega_nx*omega_ny*im_d_0 +
                                                omega_x*omega_ny*im_d_1 +
                                                omega_nx*omega_y*im_d_2 +
                                                omega_x*omega_y*im_d_3)

        return im_ud.type(torch.uint8)


# class UndistortNetwork(nn.Module):
#     def __init__(self):
#         super(UndistortNetwork, self).__init__()
#         self.undistort_layer = UndistortLayer()
#
#     def forward(self, x):
#         return x


if __name__ == "__main__":


    undistort_layer = UndistortLayer()

    # input_img = np.array([[[2, 3, 1, 0],
    #                        [4, 2, 1, 1],
    #                        [0, 3, 2, 2]],
    #                       [[5, 6, 3, 4],
    #                        [0, 1, 5, 6],
    #                        [4, 4, 3, 2]],
    #                       [[5, 2, 8, 7],
    #                        [4, 7, 8, 7],
    #                        [6, 5, 3, 3]]], dtype=np.uint8)

    input_img = cv2.imread("distorted_img_k=-0.4_dx=50_dy=-10.jpg")

    k = torch.tensor(-0.4, dtype=torch.float)
    dx = torch.tensor(50, dtype=torch.float)
    dy = torch.tensor(-10, dtype=torch.float)

    # convert input image to tensor
    input_img_m = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img_m = np.transpose(input_img_m, (2, 0, 1))
    input_tensor = torch.tensor(input_img_m, dtype=torch.float)
    output = undistort_layer(input_tensor, k, dx, dy)

    # backprop
#    v = torch.ones(3, 400, 500, dtype=torch.float)  # shape: c, h, w
#    output.backward(v)
#    print(k.grad)

    output_img = output.numpy().transpose((1, 2, 0))  # with batch dim: output.permute(0, 3, 1, 2)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    while True:
        cv2.imshow("input", input_img)
        cv2.imshow("ouput", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    #output.backward()

    print(input_tensor)
    print(output)
