# %%
from source.models import MonetCINN_112_blocks10
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
cond_net = models.ConditionNet()
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
from models import MonetCINN_112_blocks10

pretrained_path = 'C:/Users/Jonas/.cache/torch/hub'

net = MonetCINN_112_blocks10(0.01, pretrained_path)
torch.save(net.state_dict(), 'checkpoint.pt')

state_dict_loaded = torch.load('checkpoint.pt')
net_loaded = MonetCINN_112_blocks10(0.01, pretrained_path)
net_loaded.load_state_dict(state_dict_loaded)

# %%
import torch
from models import MonetCINN_112_blocks10

pretrained_path = 'C:/Users/Jonas/.cache/torch/hub'

net1 = MonetCINN_112_blocks10(0.01, pretrained_path)
print(net1.state_dict().keys())

# %%
net2 = MonetCINN_112_blocks10(0.01, pretrained_path)
print(net2.state_dict().keys())

# %%
net3 = MonetCINN_112_blocks10(0.01, pretrained_path)
print(net3.state_dict().keys())

# %%
net4 = MonetCINN_112_blocks10(0.01, pretrained_path)
print(net4.state_dict().keys())

# %%
# %%
net5 = MonetCINN_112_blocks10(0.01, pretrained_path)
print(net5.state_dict().keys())

# %%
