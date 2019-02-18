import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL.Image

import torchvision
from torchvision import datasets, models, transforms

from distortion_dataset import DistortionDataset, classes_to_parameters


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
                in range [-0.4, 0]. Expected data type is torch.float32.
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
        k = k.expand([c, h, w, -1]).permute(3, 0, 1, 2)
        dx = dx.expand([c, h, w, -1]).permute(3, 0, 1, 2)
        dy = dy.expand([c, h, w, -1]).permute(3, 0, 1, 2)

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

        return im_ud#.type(torch.uint8)



class UndistortNetwork(nn.Module):
    def __init__(self):
        super(UndistortNetwork, self).__init__()
        self.undistort_layer = UndistortLayer()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2d_5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2d_6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2d_7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv2d_8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.avg_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=65536, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, im_d):
        debug = False
        if debug: print(im_d.shape)
        x = F.relu(self.conv2d_1(im_d))
        if debug: print(x.shape)
        x = F.relu(self.conv2d_2(x))
        if debug: print(x.shape)
        x = self.avg_pooling(x)
        if debug: print(x.shape)
        x = F.relu(self.conv2d_3(x))
        if debug: print(x.shape)
        x = F.relu(self.conv2d_4(x))
        if debug: print(x.shape)
        x = self.avg_pooling(x)
        if debug: print(x.shape)
        x = F.relu(self.conv2d_5(x))
        if debug: print(x.shape)
        x = F.relu(self.conv2d_6(x))
        if debug: print(x.shape)
        x = self.avg_pooling(x)
        if debug: print(x.shape)
        x = F.relu(self.conv2d_7(x))
        if debug: print(x.shape)
        x = F.relu(self.conv2d_8(x))
        if debug: print(x.shape)
        x = self.avg_pooling(x)
        if debug: print(x.shape)

        x = x.view(-1, 65536)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))# values can be only > 0
        if debug: print(x.shape)

        # limit ranges
        k = -1*torch.clamp(x[:, 0], min=0, max=0.4)  # k will be in [-0.4 .. 0]
        dx = torch.clamp(x[:, 1], min=0, max=100)-50  # dx will be in [-50 .. 50]
        dy = torch.clamp(x[:, 2], min=0, max=100)-50  # dy will be in [-50 .. 50]

        if debug:
            print(k)
            print(dx)
            print(dy)

        im_ud = self.undistort_layer(im_d, k, dx, dy)
        return im_ud


if __name__ == "__main__":


    undistort_model = UndistortNetwork()

    input_img = cv2.imread("distorted_img_k=-0.4_dx=50_dy=-10.jpg")

    #k = torch.tensor([-0.4, -0.2], dtype=torch.float, requires_grad=True)
    #dx = torch.tensor([50, 50], dtype=torch.float, requires_grad=True)
    #dy = torch.tensor([-10, -10], dtype=torch.float, requires_grad=True)

    # convert input image to tensor
    input_img_m = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img_m = PIL.Image.fromarray(input_img_m)
    input_tensor = transforms.ToTensor()(input_img_m)
    input_tensor = input_tensor.unsqueeze(0)

    # compute feed forward
    output = undistort_model(input_tensor)

    # backprop
    #v = torch.ones(3, 400, 500, dtype=torch.float)  # shape: c, h, w
    #output.backward(v)
    #print(input_tensor.grad)

    output_img = output.numpy().transpose((0, 2, 3, 1))  # with batch dim: output.permute(0, 3, 1, 2)
    output_img1 = output_img[0, :, :, :]
    output_img2 = output_img[1, :, :, :]
    output_img1 = cv2.cvtColor(output_img1, cv2.COLOR_RGB2BGR)
    output_img2 = cv2.cvtColor(output_img2, cv2.COLOR_RGB2BGR)

    while True:
        cv2.imshow("input", input_img)
        cv2.imshow("ouput1", output_img1)
        cv2.imshow("ouput2", output_img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
