# %%
from models import MonetCINN_112_blocks10_debug
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import MonetCINN_112_blocks10, ConditionNet, MonetCINN_112_blocks10_debug

# %%
net = MonetCINN_112_blocks10_debug(0.01).cuda()
net.eval()

# %%
c_given = net.cond_net.forward(torch.rand((1, 3, 224, 224)).cuda())

# %%
N = 100
diffs = np.zeros((N, 3, 112, 112))
diffs_norm = np.zeros((N, 3, 112, 112))
for i in range(N):
    print(i)

    x = torch.rand((1, 3, 112, 112)).cuda()
    #c = torch.rand((1, 3, 224, 224)).cuda()

    #z, _ = net.forward(x, c)
    #rec_x, _ = net.reverse_sample(z, c)
    z, _ = net.forward_c_given(x, c_given)
    rec_x, _ = net.reverse_sample_c_given(z, c_given)

    diffs[i] = x[0].cpu().detach().numpy() - rec_x[0].cpu().detach().numpy()
    diffs_norm[i] = diffs[i] / x[0].cpu().detach().numpy()

# %%
plt.plot(np.max(diffs, axis=(1,2,3)), label='max')
plt.plot(np.min(diffs, axis=(1,2,3)), label='min')
plt.plot(np.mean(diffs, axis=(1,2,3)), label='mean')

plt.legend()

plt.show()

# %%
plt.plot(np.max(diffs_norm, axis=(1,2,3)), label='max')
plt.plot(np.min(diffs_norm, axis=(1,2,3)), label='min')
plt.plot(np.mean(diffs_norm, axis=(1,2,3)), label='mean')

#plt.ylim(-1, 1)
plt.legend()

plt.show()

# %%
