# %%
import torch

from models import MonetCINN_112_blocks10, ConditionNet

# %%
net = MonetCINN_112_blocks10(0.01)

# %%
x = torch.rand((3, 112, 112))
c = torch.rand((3, 224, 224))

z, _ = net.forward(x, c)

# %%
cond_net = ConditionNet()
cond_net(c)

# %%
pretrained_dict = torch.load('checkpoints/vgg11-8a719046.pth')
pretrained_keys = list(pretrained_dict.keys())

for key, param in pretrained_dict.items():
    print(key, param.shape)

# %%
import torchvision
import torch
import os

torch.hub.set_dir(os.getcwd())
torchvision.models.vgg11(pretrained=True)
