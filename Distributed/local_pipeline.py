import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from coinstac_computation import COINSTACPyNode, ComputationPhase
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

for batch_idx, (data, target) in enumerate(train_loader):
  print(batch_idx,data.shape,target.shape)
  break

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

class PhaseLoadData(ComputationPhase):
    def compute(self):
        out = {}
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        local_gradients = [param.grad.clone() for param in model.parameters()]
        #data = np.random.rand(*self.input['matrix_shape'])
        out.update(**self.send("site_matrix", local_gradients))
        #c = np.savetxt(self.out_dir + os.sep + "site_matrix.txt", data, delimiter =', ')
        return out


class PhaseSaveResult(ComputationPhase):
    def compute(self):
        data = self.recv('averaged_matrix')
        # Update the model's gradients with the aggregated gradients
        with torch.no_grad():
            for param, avg_grad in zip(model.parameters(), data):
                if param.requires_grad:
                    param.grad = avg_grad
        print('jkdfvkdvkdvkndkvnkdfvkdfv')
        #c = np.savetxt(self.out_dir + os.sep + "averaged_matrix.txt", data, delimiter =', ')
        torch.save(model.state_dict(), self.out_dir + os.sep +'model.pth')

local = COINSTACPyNode(mode='local', debug=True)
local.add_phase(PhaseLoadData)
local.add_phase(PhaseSaveResult)
