import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nibabel as nib
import pandas as pd
import numpy as np

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