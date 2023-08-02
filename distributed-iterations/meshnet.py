import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint_sequential
import os
import numpy as np
import torch


def faster_dice(x, y, labels, fudge_factor=1e-8):
    """Faster PyTorch implementation of Dice scores.
    :param x: input label map as torch.Tensor
    :param y: input label map as torch.Tensor of the same size as x
    :param labels: list of labels to evaluate on
    :param fudge_factor: an epsilon value to avoid division by zero
    :return: pytorch Tensor with Dice scores in the same order as labels.
    """

    assert x.shape == y.shape, "both inputs should have same size, had {} and {}".format(
        x.shape, y.shape
    )

    if len(labels) > 1:

        dice_score = torch.zeros(len(labels))
        for label in labels:
            x_label = x == label
            y_label = y == label
            xy_label = (x_label & y_label).sum()
            dice_score[label] = (
                2 * xy_label / (x_label.sum() + y_label.sum() + fudge_factor)
            )

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



from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class trainer:
  def __init__(self,n_channels, n_classes, trainloader, valloader, subvol_shape, epoches,modelpth,lrate=0.0007):
    self.n_channels = n_channels  # Number of input channels
    self.n_classes = n_classes # Number of output classes
    self.model = enMesh_checkpoint(self.n_channels, self.n_classes).to(device, dtype=torch.float32)
    self.criterion = nn.CrossEntropyLoss()
    self.lrate = lrate
    self.trainloader = trainloader
    self.valloader = valloader
    self.subvol_shape = subvol_shape
    self.epoches = epoches
    self.modelpth = modelpth
    self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lrate)


  def train(self):
        try:
            self.model.load_state_dict(torch.load(self.modelpth))
        except:
            print('No valid pretained model.pth file mentioned')
        self.model.train()
        for images, labels in self.trainloader:
          if 1 in torch.argmax(torch.squeeze(labels),0) or 2 in torch.argmax(torch.squeeze(labels),0):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss=self.criterion(outputs, labels)
            dice_scores = faster_dice(torch.argmax(torch.squeeze(outputs),0), torch.argmax(torch.squeeze(labels),0), labels=[0, 1, 2])  # Specify the labels to evaluate on
            loss = loss+ (1-dice_scores.mean().item())
            loss.backward()
        local_gradients = [param.grad.clone() for param in self.model.parameters()]
        return local_gradients
  
  def optimize(self,agg_grad,path):
    with torch.no_grad():
            for param, avg_grad in zip(self.model.parameters(), agg_grad):
                if param.requires_grad:
                    param.grad = avg_grad
    torch.save(self.model.state_dict(), path + os.sep +'model.pth')
    self.optimizer.step()
