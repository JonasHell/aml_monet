# %%
from source.models import MonetCINN_VGG
import torchvision.models as models

# %%
resnet18 = models.resnet18(pretrained=True)

# %%
resnet18.modules

# %%
type(resnet18.modules)

# %%
vgg11 = models.vgg11_bn(pretrained=True)

# %%
vgg11.modules

# %%
import torch
torch.hub.set_dir('D:/Dropbox/Uni/Master/2021_09_12_AdvnMachLearn/aml_monet/source')

# %%
vgg11_nobn = models.vgg11(pretrained=True)

# %%
vgg11_nobn.modules

# %%
import torch
checkpoint = torch.load('C:/Users/Jonas/.cache/torch/hub/checkpoints/vgg11-8a719046.pth')
print(type(checkpoint))
print(checkpoint.keys())
#print(checkpoint)
print(len(checkpoint))

# %%
import models
cond_net = models.ConditionNet_VGG()
print(cond_net.modules)
print('********************')
print()
counter = 0
counter2 = 0
for key, param in cond_net.named_parameters():
    print(key)
    print(param.shape)
    print()
    counter2 += 1
    if param.requires_grad:
        counter += 1

print(counter)
print(counter2)

# %%
import models
cinn = models.MonetCINN_112_blocks10(0.001)

# %%
for key, param in cinn.named_parameters():
    print(key)
    print(param)
    print()

# %%
type(cinn.named_parameters()['cinn.module_list.0.subnet1.2'])

# %%
cinn.initialize_weights()

# %%
for idx, m in enumerate(cinn.modules()):
    print(idx, '->', m)

# %%
import torch
from models import MonetCINN_VGG

pretrained_path = 'C:/Users/Jonas/.cache/torch/hub'

net = MonetCINN_VGG(0.01, pretrained_path)
torch.save(net.state_dict(), 'checkpoint.pt')

state_dict_loaded = torch.load('checkpoint.pt')
net_loaded = MonetCINN_VGG(0.01, pretrained_path)
net_loaded.load_state_dict(state_dict_loaded)

# %%
import torch
from models import MonetCINN_VGG

pretrained_path = 'C:/Users/Jonas/.cache/torch/hub'

net1 = MonetCINN_VGG(0.01, pretrained_path)
print(net1.state_dict().keys())

# %%
net2 = MonetCINN_VGG(0.01, pretrained_path)
print(net2.state_dict().keys())

# %%
net3 = MonetCINN_VGG(0.01, pretrained_path)
print(net3.state_dict().keys())

# %%
net4 = MonetCINN_VGG(0.01, pretrained_path)
print(net4.state_dict().keys())

# %%
# %%
net5 = MonetCINN_VGG(0.01, pretrained_path)
print(net5.state_dict().keys())

# %% check pretrained init
from models import ConditionNet_VGG
import torch

cond_net = ConditionNet_VGG()

pretrained_dict = torch.load('checkpoints/vgg11-8a719046.pth')
pretrained_keys = list(pretrained_dict.keys())

# initialize with pretrained weights and biases
for i, (key, param) in enumerate(cond_net.named_parameters()):
    print(key, type(param.data), param.data.shape)
    print(pretrained_keys[i], type(pretrained_dict[pretrained_keys[i]]), pretrained_dict[pretrained_keys[i]].shape)

# %%
from models import ConditionNet_debug

net = ConditionNet_debug()
for key, param in net.named_parameters():
    print(key)
    print(param.shape)
    print(param)
    print()

# %%
net.initialize_zero()
for key, param in net.named_parameters():
    print(key)
    print(param.shape)
    print(param)
    print()

# %%
import os

net.initialize_pretrained(os.getcwd())
for key, param in net.named_parameters():
    print(key)
    print(param.shape)
    print(param)
    print()

# %%
import torch

pretrained_dict = torch.load('checkpoints/vgg11-8a719046.pth')
print(pretrained_dict)

# %%
import numpy as np
import torchvision.models as models

# %%
net_dict = {}

net_dict['alexnet'] = models.alexnet()
net_dict['vgg11'] = models.vgg11()
net_dict['vgg16'] = models.vgg16()
net_dict['resnet18'] = models.resnet18()
net_dict['resnet50'] = models.resnet50()
net_dict['squeezenet'] = models.squeezenet1_0()
net_dict['squeezenet1'] = models.squeezenet1_1()
net_dict['densenet121'] = models.densenet121()
net_dict['densenet161'] = models.densenet161()
net_dict['inception'] = models.inception_v3()
net_dict['googlenet'] = models.googlenet()
net_dict['shufflenet05'] = models.shufflenet_v2_x0_5()
net_dict['shufflenet1'] = models.shufflenet_v2_x1_0()
net_dict['mobilenet_v2'] = models.mobilenet_v2()
net_dict['mobilenet_v3_large'] = models.mobilenet_v3_large()
net_dict['mobilenet_v3_small'] = models.mobilenet_v3_small()
net_dict['resnext50_32x4d'] = models.resnext50_32x4d()
net_dict['wide_resnet50_2'] = models.wide_resnet50_2()
net_dict['mnasnet05'] = models.mnasnet0_5()
net_dict['mnasnet1'] = models.mnasnet1_0()

size_dict = {}

for key, net in net_dict.items():
    num_params = 0
    for _, param in net.named_parameters():
        num_params += np.product(param.shape)
    size_dict[key] = num_params

sorted_size_dict = {k: v for k, v in sorted(size_dict.items(), key=lambda item: item[1])}

for k, v in sorted_size_dict.items():
    print(k, ': ', v)

# %%
import os
import torch
torch.hub.set_dir(os.getcwd())

# %%
import torchvision.models as models
squeeze = models.squeezenet1_1(pretrained=True)
squeeze.eval()

# %%
squeeze = models.squeezenet1_0(pretrained=True)
squeeze.eval()

# %%
squeeze

# %%
squeeze.features[0:5]

# %%
y = torch.rand((1, 3, 226, 226))
for i in range(len(squeeze.features)):
    y = squeeze.features[i].forward(y)
    print(i, y.shape)

for i in range(len(squeeze.classifier)):
    y = squeeze.classifier[i].forward(y)
    print(i, y.shape)

# %%
y = torch.rand((1, 3, 230, 230))
y = squeeze(y)
print(y.shape)

# %%
y = torch.rand((1, 3, 230, 230))
y = squeeze.features(y)
y.shape

# %%
