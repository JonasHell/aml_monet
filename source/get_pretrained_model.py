# %%
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
vgg11_nobn = models.vgg11(pretrained=True)

# %%
vgg11_nobn.modules
