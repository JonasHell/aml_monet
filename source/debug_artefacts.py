# %% imports
from models import MonetCINN_112_blocks10_debug
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import MonetCINN_112_blocks10, ConditionNet, MonetCINN_112_blocks10_debug

# %% create net
net = MonetCINN_112_blocks10_debug(0.01).cuda()
net.eval()

# %% output all layers with parameters
for key, param in net.named_parameters():
    print(key)
    print(param.shape)
    print(param)
    print()

# %% only needed when you wanna run with fixed condition
c_given = net.cond_net.forward(torch.rand((1, 3, 224, 224)).cuda())

# %% run 100 times with random tensor
N = 100
diffs = np.zeros((N, 3, 112, 112))
diffs_norm = np.zeros((N, 3, 112, 112))
for i in range(N):
    print(i)

    x = torch.rand((1, 3, 112, 112)).cuda()
    #c = torch.rand((1, 3, 224, 224)).cuda()

    #z, _ = net.forward(x, c)
    #rec_x, _ = net.reverse_sample(z, c)
    #z, _ = net.forward_c_given(x, c_given)
    #rec_x, _ = net.reverse_sample_c_given(z, c_given)
    z, _ = net.forward(x)
    rec_x, _ = net.reverse_sample(z)

    diffs[i] = x[0].cpu().detach().numpy() - rec_x[0].cpu().detach().numpy()
    diffs_norm[i] = diffs[i] / x[0].cpu().detach().numpy()

# %% compare x and z to debug identity problem
print(np.sort(x.flatten().cpu().detach().numpy()))
print(np.sort(z.flatten().cpu().detach().numpy()))

# %% compare x and z to debug identity problem
plt.plot(np.sort(x.flatten().cpu().detach().numpy()), label='x')
plt.plot(np.sort(z.flatten().cpu().detach().numpy()), label='z')
plt.legend()

# %% compare x and reconstruction to debug invertibility problem
plt.plot(np.max(diffs, axis=(1,2,3)), label='max')
plt.plot(np.min(diffs, axis=(1,2,3)), label='min')
plt.plot(np.mean(diffs, axis=(1,2,3)), label='mean')

plt.legend()

plt.show()

# %% compare x and reconstruction to debug invertibility problem (relative differences)
plt.plot(np.max(diffs_norm, axis=(1,2,3)), label='max')
plt.plot(np.min(diffs_norm, axis=(1,2,3)), label='min')
plt.plot(np.mean(diffs_norm, axis=(1,2,3)), label='mean')

#plt.ylim(-1, 1)
plt.legend()

plt.show()

# %%

# %%
