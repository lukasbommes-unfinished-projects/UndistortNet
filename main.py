import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
from tqdm import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

from distortion_dataset import DistortionDataset


# TODO:
# - save checkpoint files every K iterations
# - visualize training via visdom plots

np.random.seed(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class UndistortNet(nn.Module):
    def __init__(self):
        super(UndistortNet, self).__init__()
        base_model = models.resnet101(pretrained=True)
        # truncate base model's fully-connected layer
        modules = list(base_model.children())[:-1]
        self.base_model = nn.Sequential(*modules)
        #for param in self.base_model.parameters():
        #    param.requires_grad = True
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(in_features=73728, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=512, bias=True)
        self.fc3 = nn.Linear(in_features=512, out_features=101, bias=True)

    def forward(self, x):
        x = self.base_model(x)
        print(x.shape)
        x = x.view(-1, 73728)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        print(x.shape)
        return x


def visualize_data(dataloaders, num_images=3):
    images_shown = 0
    for x, x_edges, y in dataloaders['val']:
        for x, distortion_coefficient in zip(x, y):
            print(x.shape)
            print(y.shape)
            print("image {}: distortion: {}".format(images_shown, -4e-8*distortion_coefficient.numpy()))

            x = x.permute(1, 2, 0).numpy()
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            cv2.imshow("image_{}".format(images_shown), x)
            cv2.waitKey(5000)

            images_shown += 1

            if images_shown == num_images:
                return


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    # network setup
    model = UndistortNet()
    print(model)
    model = model.to(device)

    print("Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer, learning rate, etc.
    criterion = nn.NLLLoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # load data
    data_dir = 'dataset'
    image_datasets = {x: DistortionDataset(os.path.join(data_dir, x)) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                   batch_size=16, shuffle=True, num_workers=12)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    visualize_data(dataloaders)

    if False:

        # train network
        num_epochs = 100

        best_model_weights = copy.deepcopy(model.state_dict())
        lowest_loss = 9999.99

        t0 = time.time()

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
                for x, x_edges, y in tqdm(dataloaders[phase], desc=phase):
                    x = x.to(device)  # [32, 3, 224, 224]
                    x_edges = x_edges.to(device)
                    y = y.to(device)  # [32]

                    optimizer.zero_grad()

                    # forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        y_pred = model(x)
                        loss = criterion(y_pred, y)

                        # backprop when in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # stats
                    running_loss += loss.item() * x.size(0)

                epoch_loss = running_loss / dataset_sizes[phase]

                t_elapsed_epoch = time.time() - t0_epoch

                print("{} Loss: {:.4f}, took {:.1f} s".format(phase, epoch_loss, t_elapsed_epoch))

                # keep model if it is better than in previous epoch
                if phase == "val" and epoch_loss < lowest_loss:
                    lowest_loss = epoch_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    print("saved model in epoch {} as new best model".format(epoch))

        time_elapsed = time.time() - t0
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Lowest validation loss: {:4f}'.format(lowest_loss))

        # load the best model
        model.load_state_dict(best_model_weights)
