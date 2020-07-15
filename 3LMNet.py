import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import csv
from init_weight import *
import warnings
warnings.filterwarnings("ignore")


def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


class Dataset_3LMNet(data.Dataset):

    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            # print(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            if use_transform is not None:
                image = use_transform(image)
            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X  #[28, 3, 122, 122]

    def read_csv(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            lip_data = pd.read_csv(open(os.path.join(path, selected_folder, 'frame_{:03d}.csv'.format(i)), 'r'),header=None)
            lip_data = lip_data.as_matrix()  # juzhen
            lip_data = torch.from_numpy(lip_data)  # from juzhen to tensor double
            lip_data = lip_data.float()  # from tensor double to tensor float
            lip_data = lip_data.permute(1, 0)

            #if use_transform is not None:
            #   lip_data = use_transform(lip_data)

            X.append(lip_data)
        X = torch.stack(X, dim=0).permute(1,0,2) # [3, 28, 200])

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        # X = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        X = self.read_csv(self.data_path, folder, self.transform)
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor
        return X, y




class 3LMNet(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, num_classes=68):
        super(3LMNet, self).__init__()
        print('------ 3LMNet model-----')

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        
        ## point
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32, kernel_size=5,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        ## frame
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
        )

        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[1:-1]  # delete the last fc layer.
        # print(len(modules))  
        # print(modules)

        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, num_classes)

        # initial weight
        initial_model_weight(layers=list(self.children()))
        print('weight initial finished!')

    def forward(self, x_3d):
        # ResNet CNN
        x_1 = self.conv1_1(self.W * x_3d + x_3d)
        x_2 = self.conv1_2(x_3d)
        x = self.resnet(torch.cat((x_1,x_2),dim=1))
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x
