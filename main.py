import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
from tqdm import *

from visdom import Visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

from distortion_dataset import DistortionDataset, classes_to_parameters


# Experiments:
# - Try to do normal regression instead of binned outputs
# - change network architecture
# - try different losses
# - look at network output (what is it predicting? Visualize prediction results.)
# - play around with optimizer params
# - try end-to-end training with undistort layer

np.random.seed(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class UndistortNet(nn.Module):
    def __init__(self):
        super(UndistortNet, self).__init__()
        base_model = models.resnet50(pretrained=True)
        # truncate base model's fully-connected and avg pool layer
        modules = list(base_model.children())[:-2]
        self.base_model = nn.Sequential(*modules)
        #for param in self.base_model.parameters():
        #    param.requires_grad = True

        self.rawim_conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

        # tranpose convolution for feature upsampling
        self.deconv2d_1 = nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=2, stride=2)
        self.deconv2d_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.deconv2d_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.deconv2d_4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.deconv2d_5 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2)

        # distortion parameter estimation network
        self.dpen_conv2d_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.dpen_conv2d_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.dpen_conv2d_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=2)
        self.dpen_conv2d_7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.dpen_conv2d_9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2)
        self.dpen_pooling2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dpen_batch_norm_1 = nn.BatchNorm2d(num_features=64)
        self.dpen_batch_norm_2 = nn.BatchNorm2d(num_features=64)
        self.dpen_batch_norm_3 = nn.BatchNorm2d(num_features=128)
        self.dpen_batch_norm_4 = nn.BatchNorm2d(num_features=128)
        self.dpen_batch_norm_5 = nn.BatchNorm2d(num_features=256)
        self.dpen_batch_norm_6 = nn.BatchNorm2d(num_features=256)
        self.dpen_batch_norm_7 = nn.BatchNorm2d(num_features=512)
        self.dpen_batch_norm_8 = nn.BatchNorm2d(num_features=512)
        self.dpen_batch_norm_9 = nn.BatchNorm2d(num_features=512)
        self.dpen_batch_norm_10 = nn.BatchNorm2d(num_features=512)
        self.dropout = nn.Dropout(p=0.5)

        # linear output layers
        self.fc1 = nn.Linear(in_features=8192, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc_k = nn.Linear(in_features=1024, out_features=21)
        self.fc_dx = nn.Linear(in_features=1024, out_features=21)
        self.fc_dy = nn.Linear(in_features=1024, out_features=21)

    def forward(self, x):
        debug = False
        if debug: print("input: ", x.shape)

        # 3 x 3 conv layer on raw input image
        x_raw_image = self.rawim_conv2d(x)
        if debug: print("x_raw_image conv", x_raw_image.shape)

        # extract features with pretrained base network (TODO: take out features of lower layers)
        x_features = self.base_model(x)
        if debug: print("base network: ", x_features.shape)

        # deconvolution to rise spatial resolution of extracted features
        x_features = self.deconv2d_1(x_features)
        if debug: print("base deconv 1: ", x_features.shape)
        x_features = self.deconv2d_2(x_features)
        if debug: print("base deconv 2: ", x_features.shape)
        x_features = self.deconv2d_3(x_features)
        if debug: print("base deconv 3: ", x_features.shape)
        x_features = self.deconv2d_4(x_features)
        if debug: print("base deconv 4: ", x_features.shape)
        x_features = self.deconv2d_5(x_features)
        if debug: print("base deconv 5: ", x_features.shape)

        # concatenate raw image features and upsampled base features
        x = torch.cat((x_raw_image, x_features), 1)
        if debug: print("after concat: ", x.shape)

        # distortion parameter estimation network
        # --------- L1 (Conv 1) ---------
        x = self.dpen_conv2d_1(x)
        if debug: print("conv 1:", x.shape)
        x = self.dpen_batch_norm_1(x)
        x = F.relu(x)
        # --------- L2 (Conv 2) ---------
        x = self.dpen_conv2d_2(x)
        if debug: print("conv 2:", x.shape)
        x = self.dpen_batch_norm_2(x)
        x = F.relu(x)
        # --------- L3 (Conv 3) ---------
        x = self.dpen_conv2d_3(x)
        if debug: print("conv 3:", x.shape)
        x = self.dpen_batch_norm_3(x)
        x = F.relu(x)
        # --------- L4 (Conv 4) ---------
        x = self.dpen_conv2d_4(x)
        if debug: print("conv 4:", x.shape)
        x = self.dpen_batch_norm_4(x)
        x = F.relu(x)
        # --------- L5 (Conv 5) ---------
        x = self.dpen_conv2d_5(x)
        if debug: print("conv 5:", x.shape)
        x = self.dpen_batch_norm_5(x)
        x = F.relu(x)
        # --------- L6 (Conv 6) ---------
        x = self.dpen_conv2d_6(x)
        if debug: print("conv 6:", x.shape)
        x = self.dpen_batch_norm_6(x)
        x = F.relu(x)
        # --------- L7 (Conv 7) ---------
        x = self.dpen_conv2d_7(x)
        if debug: print("conv 7:", x.shape)
        x = self.dpen_batch_norm_7(x)
        x = F.relu(x)
        # --------- L8 (Conv 8) ---------
        x = self.dpen_conv2d_8(x)
        if debug: print("conv 8:", x.shape)
        x = self.dpen_batch_norm_8(x)
        # x = F.relu(x)
        # --------- L9 (Conv 9) ---------
        x = self.dpen_conv2d_9(x)
        if debug: print("conv 9:", x.shape)
        x = self.dpen_batch_norm_9(x)
        x = F.relu(x)
        # --------- L10 (Conv 10) ---------
        x = self.dpen_conv2d_10(x)
        if debug: print("conv 10:", x.shape)
        x = self.dpen_batch_norm_10(x)
        x = F.relu(x)
        x = self.dpen_pooling2d(x)
        if debug: print("pooling:", x.shape)
        x = self.dropout(x)

        # linear output layers
        # --------- FC 1 ---------
        x = x.view(x.size(0), -1)
        #if self.fc1 is None:
        #    self.fc1 = nn.Linear(in_features=x.size(1), out_features=1024)
        #    self.fc1 = self.fc1.to(device)
        x = self.fc1(x)
        if debug: print("FC 1: ", x.shape)
        x = F.relu(x)
        x = self.dropout(x)
        # --------- FC 2 ---------
        x = self.fc2(x)
        if debug: print("FC 2: ", x.shape)
        x = F.relu(x)
        x = self.dropout(x)
        # --------- FC k ---------
        k = self.fc_k(x)
        k = F.log_softmax(k, dim=1)
        if debug: print("FC k: ", k.shape)
        # --------- FC dx ---------
        dx = self.fc_dx(x)
        dx = F.log_softmax(dx, dim=1)
        if debug: print("FC dx: ", dx.shape)
        # --------- FC dy ---------
        dy = self.fc_dy(x)
        dy = F.log_softmax(dy, dim=1)
        if debug: print("FC dy: ", dy.shape)

        return k, dx, dy


def _convert_to_opencv(image):
    image = image.permute(1, 2, 0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def visualize_data(dataloaders, num_images=3, phase='train'):
    images_shown = 0
    for data in tqdm(dataloaders[phase], desc=phase):
        for im_d, im_d_c, im_ud, k, dx, dy in zip(*data):

            k = k.numpy()
            dx = dx.numpy()
            dy = dy.numpy()
            k = -4e-3*k
            print("image {}: k = {}, dx = {}, dy = {}".format(images_shown, k, dx, dy))

            im_d = _convert_to_opencv(im_d)
            im_ud = _convert_to_opencv(im_ud)
            im_d_c = _convert_to_opencv(im_d_c)

            while True:
                cv2.imshow("image_d{}".format(images_shown), im_d)
                cv2.imshow("image_d_c{}".format(images_shown), im_d_c)
                cv2.imshow("image_ud{}".format(images_shown), im_ud)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            images_shown += 1

            if images_shown == num_images:
                return


if __name__ == "__main__":

    # for visualization
    vis = Visdom(server="http://undistort_net_visdom", port=8097)
    print("Visualizing network training via Visdom. Connect to http://localhost:8097 to view stats.")

    # model setup, training and validation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # network setup (run on both GPUs)
    model = UndistortNet()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    print(model)
    model = model.to(device)

    print("Number of trainable parameters: {}".format(count_parameters(model)))

    # define losses
    loss_k_criterion = nn.NLLLoss(reduction="mean")
    loss_dx_criterion = nn.NLLLoss(reduction="mean")
    loss_dy_criterion = nn.NLLLoss(reduction="mean")

    # optimizer, learning rate, etc.
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # 1e-3
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # load data
    data_dir = 'dataset'
    image_datasets = {x: DistortionDataset(os.path.join(data_dir, x)) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                   batch_size=64, shuffle=True, num_workers=12)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    #visualize_data(dataloaders)

    if True:

        # train network
        num_epochs = 10
        lowest_loss = 9999.99

        t0 = time.time()

        win = vis.line([0], [-1], opts=dict(title="Losses", xlabel="Step", ylabel="Loss"))

        for epoch in range(num_epochs):
            t0_epoch = time.time()
            print("--------------------")
            print("Epoch {}/{}".format(epoch+1, num_epochs))

            for phase in ["train", "val"]:
                if phase == "train":
                    exp_lr_scheduler.step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                # iterate over data
                for step, data in enumerate(dataloaders[phase]):
                    im_d, im_d_c, im_ud, k, dx, dy = data

                    # move to GPU
                    im_d = im_d.to(device)  # [B, 3, 512, 512]
                    im_d_c = im_d_c.to(device)
                    im_ud = im_ud.to(device)
                    k = k.to(device)  # [B]
                    dx = dx.to(device)
                    dy = dy.to(device)

                    optimizer.zero_grad()

                    # forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        k_pred, dx_pred, dy_pred = model(im_d_c)
                        loss_k = loss_k_criterion(k_pred, k)
                        loss_dx = loss_k_criterion(dx_pred, dx)
                        loss_dy = loss_k_criterion(dy_pred, dy)
                        loss = loss_k + loss_dx + loss_dy

                        # backprop when in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # stats
                    running_loss += loss.item() * im_d_c.size(0)

                    # report every 10 steps
                    if step % 10 == 0:
                        print("step: {}, loss: {}".format(step, loss.item()))
                        vis.line([loss.item()], [step], win=win, update='append')

                epoch_loss = running_loss / dataset_sizes[phase]

                t_elapsed_epoch = time.time() - t0_epoch

                print("{} Loss: {:.4f}, took {:.1f} s".format(phase, epoch_loss, t_elapsed_epoch))

                # create and safe checkpoint
                checkpoint = {
                    "epoch": epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }
                torch.save(checkpoint, "model/checkpoints/checkpoint_epoch_{}.tar".format(epoch))
                print("Saved checkpoint.")

                # safe separate checkpoint if model is better than in previous epoch
                if phase == "val" and epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    torch.save(checkpoint, "model/checkpoints/best_model.tar")
                    print("Saved model in epoch {} as new best model.".format(epoch))

        time_elapsed = time.time() - t0
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Lowest validation loss: {:4f}'.format(lowest_loss))

        # load the best model
        model.load_state_dict(best_model_weights)
