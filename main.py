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
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

from distortion_dataset import DistortionDataset


def visualize_data(dataloaders, num_images=3):
    images_shown = 0
    for i, (x, y) in enumerate(dataloaders['val']):
        # convert to opencv format
        for image in x:
            image = image.permute(1, 2, 0).numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            cv2.imshow("image_{}".format(i), image)
            cv2.waitKey(500)

            images_shown += 1

            if images_shown==num_images:
                return


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # network setup
    model = models.resnet18(pretrained=True)
    model = model.to(device)

    # optimizer, learning rate, etc.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/3)

    # load data
    data_dir = "hymenoptera_data"
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_dir = 'imagenet'
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']}
    image_datasets = {x: DistortionDataset(os.path.join(data_dir, x),
                                                    data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                   batch_size=32, shuffle=True, num_workers=12)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    visualize_data(dataloaders)

    # # train network
    # num_epochs = 2
    #
    # best_model_weights = copy.deepcopy(model.state_dict())
    # lowest_loss = 9999.99
    #
    # t0 = time.time()
    #
    # for epoch in range(num_epochs):
    #     t0_epoch = time.time()
    #     print("--------------------")
    #     print("Epoch {}/{}".format(epoch+1, num_epochs))
    #
    #     for phase in ["train", "val"]:
    #         if phase == "train":
    #             exp_lr_scheduler.step()
    #             model.train()
    #         else:
    #             model.eval()
    #
    #         running_loss = 0.0
    #
    #         # iterate over data
    #         for x, y in tqdm(dataloaders[phase], desc=phase):
    #             x = x.to(device)  # [32, 3, 224, 224]
    #             y = y.to(device)  # [32]
    #
    #             optimizer.zero_grad()
    #
    #             # forward pass
    #             with torch.set_grad_enabled(phase == "train"):
    #                 y_predicted = model(x)
    #                 #_, y_class = torch.max(y_predicted, 1)
    #                 loss = criterion(y_predicted, y)
    #
    #                 # backprop when in training phase
    #                 if phase == "train":
    #                     loss.backward()
    #                     optimizer.step()
    #
    #             # stats
    #             running_loss += loss.item() * x.size(0)
    #
    #         epoch_loss = running_loss / dataset_sizes[phase]
    #
    #         t_elapsed_epoch = time.time() - t0_epoch
    #
    #         print("{} Loss: {:.4f}, took {:.1f} s".format(phase, epoch_loss, t_elapsed_epoch))
    #
    #         # keep model if it is better than in previous epoch
    #         if phase == "val" and epoch_loss < lowest_loss:
    #             lowest_loss = epoch_loss
    #             best_model_weights = copy.deepcopy(model.state_dict())
    #
    # time_elapsed = time.time() - t0
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Lowest validation loss: {:4f}'.format(lowest_loss))
    #
    # # load the best model
    # model.load_state_dict(best_model_weights)
