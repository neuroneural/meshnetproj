import os
import torch
import time
import requests
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential

class CubeDivider:
    def __init__(self, tensor, num_cubes):
        self.tensor = tensor
        self.num_cubes = num_cubes
        self.sub_cube_size = tensor.shape[0] // num_cubes  # Assuming the tensor is a cube

    def divide_into_sub_cubes(self):
        sub_cubes = []

        for i in range(self.num_cubes):
            for j in range(self.num_cubes):
                for k in range(self.num_cubes):
                    sub_cube = self.tensor[
                        i * self.sub_cube_size: (i + 1) * self.sub_cube_size,
                        j * self.sub_cube_size: (j + 1) * self.sub_cube_size,
                        k * self.sub_cube_size: (k + 1) * self.sub_cube_size
                    ].clone()
                    sub_cubes.append(sub_cube)

        sub_cubes = torch.stack(sub_cubes,0)
        return sub_cubes

    @staticmethod
    def reassemble_sub_cubes(sub_cubes):
        sub_cubes = torch.unbind(sub_cubes, dim=0)
        num_cubes = int(len(sub_cubes) ** (1/3))
        sub_cube_size = sub_cubes[0].shape[0]
        tensor_size = num_cubes * sub_cube_size
        tensor = torch.zeros((tensor_size, tensor_size, tensor_size), dtype=torch.float32)

        for i in range(num_cubes):
            for j in range(num_cubes):
                for k in range(num_cubes):
                    sub_cube = sub_cubes[i * num_cubes**2 + j * num_cubes + k]
                    tensor[
                        i * sub_cube_size: (i + 1) * sub_cube_size,
                        j * sub_cube_size: (j + 1) * sub_cube_size,
                        k * sub_cube_size: (k + 1) * sub_cube_size
                    ] = sub_cube

        return tensor

class DataLoaderClass:
  def __init__(self,csv_file, coor_factor, batch_size, path, count):
    self.csv_file=csv_file
    self.coor_factor=coor_factor
    self.batch_size=batch_size
    self.path = path
    self.count = count

  def dataloader(self):
    data = pd.read_csv(self.csv_file)
    volume_shape = [256, 256, 256]
    images =()
    labels=()
    iter = 0 
    for image,label in zip(data['images'],data['GWlabels']):
      if iter == self.count:
        img = nib.load(self.path+'/'+image)
        img = img.get_fdata()
        temp= np.zeros(volume_shape)
        temp[: img.shape[0], : img.shape[1], : img.shape[2]] = img
        temp = np.array(temp)
        image_data = (temp - temp.mean()) / temp.std()
        sub_temp = CubeDivider(torch.tensor(image_data),self.coor_factor)
        images = images+(sub_temp.divide_into_sub_cubes(),)

        lab = nib.load(self.path+'/'+label)
        lab = lab.get_fdata()
        temp= np.zeros(volume_shape)
        temp[: lab.shape[0], : lab.shape[1], : lab.shape[2]] = lab
        temp = np.array(temp)
        sub_temp = CubeDivider(torch.tensor(temp),self.coor_factor)
        labels = labels+(sub_temp.divide_into_sub_cubes(),)
        break
      else:
        iter = iter+1
        continue

    images = torch.stack(images)
    labels = torch.stack(labels)
    images = images.reshape(-1,1,int(volume_shape[0]/self.coor_factor),int(volume_shape[1]/self.coor_factor),int(volume_shape[2]/self.coor_factor)).float()
    labels = labels.reshape(-1,1,int(volume_shape[0]/self.coor_factor),int(volume_shape[1]/self.coor_factor),int(volume_shape[2]/self.coor_factor))
    new_labels = ()
    for temp in labels:
      new_temp = ()
      for i in [0,1,2]:
        new_temp=new_temp+ (torch.mul(torch.tensor(np.asarray(temp == i, dtype=np.float64)),1),)
      new_temp = torch.stack(new_temp)
      new_labels = new_labels + (new_temp,)
    labels = torch.stack(new_labels)
    labels = labels.reshape(-1,3,int(volume_shape[0]/self.coor_factor),int(volume_shape[1]/self.coor_factor),int(volume_shape[2]/self.coor_factor))
    dataset = torch.utils.data.TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

def remove_file(file_path):
    try:
        # Use os.remove() to delete the file
        os.remove(file_path)
        print(f"File '{file_path}' has been removed successfully.")
    except Exception as e:
        print(f"Error occurred while removing the file: {e}")

def unzip_file(zip_file_path):
    try:
        # Use the 'unzip' command to extract the contents of the zip file to the same path
        os.system(f'unzip {zip_file_path} -d {os.path.dirname(zip_file_path)}')
        print(f"Successfully extracted files from: {zip_file_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

def download_url_contents(url, folder_path, file_name):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            os.makedirs(folder_path, exist_ok=True)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Download successful. File saved at: {file_path}")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error occurred: {e}")

def get_data(path):
        download_url_contents('https://meshnet-pr-dataset.s3.amazonaws.com/data-20-1.zip',path,'data.zip')
        time.sleep(5)
        unzip_file(os.path.join(path,'data.zip'))
        time.sleep(5)
        remove_file(os.path.join(path,'data.zip'))


def faster_dice(x, y, labels, fudge_factor=1e-8):
    x = torch.argmax(torch.squeeze(x),0)
    assert x.shape == y.shape, "both inputs should have same size, had {} and {}".format(x.shape, y.shape)
    if len(labels) > 1:
        dice_score = torch.zeros(len(labels))
        for label in labels:
            x_label = x == label
            y_label = y == label
            xy_label = (x_label & y_label).sum()
            dice_score[label] = (2 * xy_label / (x_label.sum() + y_label.sum() + fudge_factor))
    else:
        dice_score = dice(x == labels[0], y == labels[0], fudge_factor=fudge_factor)
    return dice_score


def dice(x, y, fudge_factor=1e-8):
    """Implementation of dice scores ofr 0/1 numy array"""
    return 2 * torch.sum(x * y) / (torch.sum(x) + torch.sum(y) + fudge_factor)

MeshNet_5_ae16 = [
    {"in_channels": -1,"kernel_size": 3,"out_channels": 5,"padding": 1,"stride": 1,"dilation": 1,},
    {"in_channels": 5,"kernel_size": 3,"out_channels": 5,"padding": 2,"stride": 1,"dilation": 2,},
    {"in_channels": 5,"kernel_size": 3,"out_channels": 5,"padding": 4,"stride": 1,"dilation": 4,},
    {"in_channels": 5,"kernel_size": 3,"out_channels": 5,"padding": 8,"stride": 1,"dilation": 8,},
    {"in_channels": 5,"kernel_size": 3,"out_channels": 5,"padding": 16,"stride": 1,"dilation": 16,},
    {"in_channels": 5,"kernel_size": 3,"out_channels": 5,"padding": 8,"stride": 1,"dilation": 8,},
    {"in_channels": 5,"kernel_size": 3,"out_channels": 5,"padding": 4,"stride": 1,"dilation": 4,},
    {"in_channels": 5,"kernel_size": 3,"out_channels": 5,"padding": 2,"stride": 1,"dilation": 2,},
    {"in_channels": 5,"kernel_size": 3,"out_channels": 5,"padding": 1,"stride": 1,"dilation": 1,},
    {"in_channels": 5,"kernel_size": 1,"out_channels": -1,"padding": 0,"stride": 1,"dilation": 1,},
]

def ae16channels(channels=5, basearch=MeshNet_5_ae16):
    start = {"out_channels": channels}
    middle = {"in_channels": channels,"out_channels": channels}
    end = {"in_channels": channels}
    modifier = [start] + [middle for _ in range(len(basearch)-2)] + [end]
    newarch = basearch.copy()
    [x.update(y) for x,y in zip(newarch, modifier)]
    return newarch

def conv_w_bn_before_act(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
    """Configurable Conv block with Batchnorm and Dropout"""
    sequence = [("conv", nn.Conv3d(*args, **kwargs))]
    if bnorm:
        sequence.append(("bnorm", nn.BatchNorm3d(kwargs["out_channels"])))
    if gelu:
        sequence.append(("gelu", nn.GELU()))
    else:
        sequence.append(("relu", nn.ReLU(inplace=True)))
    sequence.append(("dropout", nn.Dropout3d(dropout_p)))
    layer = nn.Sequential(OrderedDict(sequence))
    return layer

def init_weights(model):
    """Set weights to be xavier normal for all Convs"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.0)

class MeshNet(nn.Module):
    """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""

    def __init__(self, n_channels, n_classes, large=True, bnorm=True, gelu=False, dropout_p=0):
        """Init"""
        if large:
            params  = ae16channels(5)
        else:
            params = MeshNet_5_ae16

        super(MeshNet, self).__init__()
        params[0]["in_channels"] = n_channels
        params[-1]["out_channels"] = n_classes
        layers = [
            conv_w_bn_before_act(dropout_p=dropout_p, bnorm=bnorm, gelu=gelu, **block_kwargs)
            for block_kwargs in params[:-1]
        ]
        layers.append(nn.Conv3d(**params[-1]))
        self.model = nn.Sequential(*layers)
        init_weights(self.model)

    def forward(self, x):
        """Forward pass"""
        x = self.model(x)
        return x

class enMesh_checkpoint(MeshNet):
    def train_forward(self, x):
        y = x
        y.requires_grad_()
        y = checkpoint_sequential(
            self.model, len(self.model), y, preserve_rng_state=False
        )
        return y

    def eval_forward(self, x):
        """Forward pass"""
        self.model.eval()
        with torch.inference_mode():
            x = self.model(x)
        return x

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

