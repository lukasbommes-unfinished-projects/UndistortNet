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

from distortion_dataset import DistortionDataset#, distortion_params
from undistort_layer import UndistortLayer

# TODO:
# - log losses to csv file for later plotting

# Experiments:
# - change network architecture
# - try different losses
# - look at network output (what is it predicting? Visualize prediction results.)
# - play around with optimizer params
# - crop and translate input image and undistorted image randomly (simply add it at the end of the pipeline)

np.random.seed(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class UndistortNet(nn.Module):
    def __init__(self):
        super(UndistortNet, self).__init__()
        base_model = models.resnet50(pretrained=True)  # try ResNet 101
        # truncate base model's fully-connected and avg pool layer
        modules = list(base_model.children())[:-2]
        self.base_model = nn.Sequential(*modules)
        #for param in self.base_model.parameters():
        #    param.requires_grad = True

        self.rawim_conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

        # undistortion layer
        self.undistort_layer = UndistortLayer()

        # tranpose convolution for feature upsampling
        self.deconv2d_1 = nn.ConvTranspose2d(in_channels=2048, out_channels=512, kernel_size=2, stride=2)
        self.deconv2d_2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.deconv2d_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.deconv2d_4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.deconv2d_5 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=2, stride=2)

        # distortion parameter estimation network
        self.dpen_conv2d_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.dpen_conv2d_8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.dpen_pooling2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dpen_batch_norm_1 = nn.BatchNorm2d(num_features=64)
        self.dpen_batch_norm_2 = nn.BatchNorm2d(num_features=64)
        self.dpen_batch_norm_3 = nn.BatchNorm2d(num_features=128)
        self.dpen_batch_norm_4 = nn.BatchNorm2d(num_features=128)
        self.dpen_batch_norm_5 = nn.BatchNorm2d(num_features=256)
        self.dpen_batch_norm_6 = nn.BatchNorm2d(num_features=256)
        self.dpen_batch_norm_7 = nn.BatchNorm2d(num_features=512)
        self.dpen_batch_norm_8 = nn.BatchNorm2d(num_features=512)
        self.dropout = nn.Dropout(p=0.5)

        # linear output layers
        self.fc1 = nn.Linear(in_features=131072, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc_k = nn.Linear(in_features=1024, out_features=1)
        self.fc_dx = nn.Linear(in_features=1024, out_features=1)
        self.fc_dy = nn.Linear(in_features=1024, out_features=1)

    def forward(self, im_d):
        if debug: print("input: ", im_d.shape)

        # 3 x 3 conv layer on raw input image
        x_raw_image = self.rawim_conv2d(im_d)
        if debug: print("x_raw_image conv", x_raw_image.shape)

        # extract features with pretrained base network (TODO: take out features of lower layers)
        x_features = self.base_model(im_d)
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
        x = self.dpen_pooling2d(x)
        if debug: print("pool 1:", x.shape)
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
        x = self.dpen_pooling2d(x)
        if debug: print("pool 2:", x.shape)
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
        x = self.dpen_pooling2d(x)
        if debug: print("pool 3:", x.shape)
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
        x = self.dpen_pooling2d(x)
        if debug: print("pool 4:", x.shape)
        x = F.relu(x)
        x = self.dropout(x)

        # linear output layers
        # --------- FC 1 ---------
        x = x.view(x.size(0), -1)
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
        k = -0.4 * F.sigmoid(k) - 0.001
        if debug: print("FC k: ", k.shape)
        # --------- FC dx ---------
        dx = self.fc_dx(x)
        dx = torch.tanh(dx)*50
        if debug: print("FC dx: ", dx.shape)
        # --------- FC dy ---------
        dy = self.fc_dy(x)
        dy = torch.tanh(dy)*50
        if debug: print("FC dy: ", dy.shape)

        print("k_pred", k.view(-1))
        print("dx_pred", dx.view(-1))
        print("dy_pred", dy.view(-1))

        # undistort input image with estimated parameters
        im_ud = self.undistort_layer(im_d, k, dx, dy)

        return im_ud


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


def evaluate_model(model, val_data_loader, val_dataset_len):
    was_training = model.training
    if was_training:
        model.eval()
    running_loss = 0.0
    for data in tqdm(val_data_loader, desc="val"):
        im_d, im_d_c, im_ud, k, dx, dy = data
        im_d, im_d_c, im_ud = im_d.to(device), im_d_c.to(device), im_ud.to(device)
        k, dx, dy = k.to(device), dx.to(device), dy.to(device)
        # forward pass
        with torch.no_grad():
            im_ud_pred = model(im_d)
            loss = loss_criterion(im_ud_pred, im_ud)
        running_loss += loss.item() * im_d.size(0)
    val_loss = running_loss / val_dataset_len
    if was_training:
        model.train()
    return val_loss


if __name__ == "__main__":

    debug = False

    # for visualization
    vis = Visdom(server="http://undistort_net_visdom", port=8097)
    print("Visualizing network training via Visdom. Connect to http://localhost:8097 to view stats.")

    # model setup, training and validation
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        num_devices = torch.cuda.device_count()
        torch.cuda.empty_cache()
    else:
        raise NotImplementedError("Currently only GPU training is supported.")

    # network setup (run on both GPUs)
    model = UndistortNet()
    model = torch.nn.DataParallel(model)
    if debug: print(model)
    model = model.to(device)

    print("Number of trainable parameters: {}".format(count_parameters(model)))

    # define losses
    loss_criterion = nn.MSELoss(reduction="sum")

    # optimizer, learning rate, etc.
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 1e-3
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # load data
    data_dir = 'dataset'
    image_datasets = {x: DistortionDataset(os.path.join(data_dir, x)) for x in ['train', 'val']}
    def worker_init(worker_id):
        np.random.seed(worker_id)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                   batch_size=32, shuffle=True, num_workers=8, worker_init_fn=worker_init)
                   for x in ['train', 'val']}

    print("Number of training images: {}".format(len(image_datasets["train"])))
    print("Number of validation images: {}".format(len(image_datasets["val"])))

#visualize_data(dataloaders)

    # train network
    num_epochs = 10
    print_log_every_steps = 10  # None to disable
    evaluate_model_every_steps = 100  # None to disable
    save_model_ckpt_every_steps = 1000  # None to disable
    save_best_model = True  # keep the best model

    if evaluate_model_every_steps:
        assert evaluate_model_every_steps % print_log_every_steps == 0, \
               "evaluate_model_every_steps should be a multiple of print_log_every_steps"

    t0 = time.time()
    validated = False

    win = vis.line([0], [-1], opts=dict(title="Losses", xlabel="Step", ylabel="Loss"))

    try:

        for epoch in range(num_epochs):
            step = 0
            num_steps_in_epoch = int(np.ceil(len(image_datasets["train"]) / dataloaders["train"].batch_size))
            t0_epoch = time.time()
            val_loss = float("inf")
            val_loss_lowest = float("inf")
            print("--------------------")
            print("Epoch {}/{}".format(epoch+1, num_epochs))

            scheduler.step()
            model.train()

            # iterate over data
            for data in dataloaders["train"]:
                im_d, im_d_c, im_ud, k, dx, dy = data
                im_d, im_d_c, im_ud = im_d.to(device), im_d_c.to(device), im_ud.to(device)
                k, dx, dy = k.to(device), dx.to(device), dy.to(device)

                # debug only
                print("k_actual", k)
                print("dx_actual", dx)
                print("dy_actual", dy)

                optimizer.zero_grad()

                # forward pass
                with torch.enable_grad():
                    im_ud_pred = model(im_d)
                    #cv2.imshow("im_d", _convert_to_opencv(im_d[0, :, :, :].detach().cpu()))
                    #cv2.imshow("im_ud", _convert_to_opencv(im_ud[0, :, :, :].detach().cpu()))
                    #cv2.waitKey(1)
                    loss = loss_criterion(im_ud_pred, im_ud)
                    # backprop
                    loss.backward()
                    optimizer.step()

                # evaluate model every "evaluate_model_every_steps" steps
                if print_log_every_steps and evaluate_model_every_steps:
                    if step > 0 and step % evaluate_model_every_steps == 0:
                        print("Running model evaluation...")
                        val_loss = evaluate_model(model, dataloaders["val"], len(image_datasets["val"]))
                        validated = True

                # report every "print_log_every_steps" steps
                if print_log_every_steps:
                    if step % print_log_every_steps == 0:
                        if validated:
                            validated = False
                            disp = [epoch+1, num_epochs, step+1, num_steps_in_epoch, time.time() - t0_epoch, loss.item(), val_loss]
                            print("epoch: {:02d}/{:02d}, step: {:06d}/{:06d}, elapsed: {:011.3f} s, train loss: {:.3f}, val loss: {:.3f}".format(*disp))
                            vis.line([val_loss], [num_steps_in_epoch * epoch + step], win=win, name='val', update='append')
                        else:
                            disp = [epoch+1, num_epochs, step+1, num_steps_in_epoch, time.time() - t0_epoch, loss.item()]
                            print("epoch: {:02d}/{:02d}, step: {:06d}/{:06d}, elapsed: {:011.3f} s, train loss: {:.3f}, val loss: ---".format(*disp))
                            vis.line([loss.item()], [num_steps_in_epoch * epoch + step], win=win, name='train', update='append')

                # save checkpoint every "save_model_ckpt_every_steps" steps
                if save_model_ckpt_every_steps:
                    if step > 0 and step % save_model_ckpt_every_steps == 0:
                        checkpoint = {
                            "epoch": epoch,
                            "step": step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': loss
                        }
                        torch.save(checkpoint, "model/checkpoints/checkpoint_epoch_{}_step_{}.tar".format(epoch, step))
                        print("Saved checkpoint.")


                # save model checkpoint if validation loss is smaller than in previous run
                if save_best_model and evaluate_model_every_steps:
                    if step > 0 and val_loss < val_loss_lowest:
                        val_loss_lowest = val_loss
                        checkpoint = {
                            "epoch": epoch,
                            "step": step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': loss
                        }
                        torch.save(checkpoint, "model/checkpoints/best_model.tar".format(epoch, step))
                        print("Saved model as new best model.")

                step += 1

    # save model when user hits CRTL + C then exit
    except KeyboardInterrupt:
        checkpoint = {
            "epoch": epoch,
            "step": step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss
        }
        torch.save(checkpoint, "model/checkpoints/model_at_exit.tar".format(epoch, step))
        print("Saving last state before exiting...")
