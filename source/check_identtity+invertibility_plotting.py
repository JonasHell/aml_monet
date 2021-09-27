# %% imports
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import *

# %% create net
net = MonetCINN_simple(0.01)#.cuda()
net.eval()

# %%
cond_net = ConditionNet_squeeze()
c = torch.rand((1, 3, 226, 226))
cond_net(c)

# %% output all layers with parameters
for key, param in net.named_parameters():
    print(key)
    print(param.shape)
    print(param)
    print()

# %% only needed when you wanna run with fixed condition
c_given = net.cond_net.forward(torch.rand((1, 3, 224, 224)).cuda())

# %% run 100 times with random tensor
N = 10
diffs = np.zeros((N, 3, 112, 112))
diffs_norm = np.zeros((N, 3, 112, 112))
#diffs = np.zeros((N, 3, 224, 224))
#diffs_norm = np.zeros((N, 3, 224, 224))

for i in range(N):
    print(i)

    x = torch.rand((1, 3, 112, 112))#.cuda()
    #c = torch.rand((1, 3, 224, 224)).cuda()
    #x = torch.rand((1, 3, 224, 224))#.cuda()
    #c = torch.rand((1, 3, 226, 226))#.cuda()

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

# %%

# %%

# %% identity and invertibility plots
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import *

# %% create net
net = MonetCINN_squeeze(0.01)
net_t = MonetCINN_squeeze(0.01)

new_state_dict = torch.load('checkpoints/monet_cinn.pt')
old_state_dict = net_t.state_dict()

j_old = -1
j_new = -1

for i in range(28, 33):
#print(f"Current permutation index {i}")
    key = f"cinn.module_list.{i}.perm"
    if key in old_state_dict:
        j_old = i
    if key in new_state_dict:
        j_new = i

print(j_old, j_new)
assert(j_old != -1)
assert(j_new != -1)

if j_old != j_new:
    new_state_dict[f"cinn.module_list.{j_old}.perm"]     = new_state_dict[f"cinn.module_list.{j_new}.perm"]
    new_state_dict[f"cinn.module_list.{j_old}.perm_inv"] = new_state_dict[f"cinn.module_list.{j_new}.perm_inv"]
    del new_state_dict[f"cinn.module_list.{j_new}.perm"]
    del new_state_dict[f"cinn.module_list.{j_new}.perm_inv"]

net_t.load_state_dict(new_state_dict)

net.eval()
net_t.eval()

# %% run 100 times with random tensor
N = 100

X = torch.rand((N, 1, 3, 224, 224))
C = torch.rand((N, 1, 3, 226, 226))

Z = np.zeros((N, 150528))
X_rec = np.zeros((N, 3, 224, 224))

Z_t = np.zeros((N, 150528))
X_rec_t = np.zeros((N, 3, 224, 224))

for i, (x, c) in enumerate(zip(X, C)):
    print(i)

    z, _ = net.forward(x, c)
    x_rec, _ = net.reverse_sample(z, c)
    Z[i] = z.detach().numpy()
    X_rec[i] = x_rec.detach().numpy()

    z, _ = net_t.forward(x, c)
    x_rec, _ = net_t.reverse_sample(z, c)
    Z_t[i] = z.detach().numpy()
    X_rec_t[i] = x_rec.detach().numpy()

X = X.detach().numpy().squeeze()
C = C.detach().numpy().squeeze()

# %% compare x and z to debug identity problem
plt.plot(np.square(np.sort(X.reshape(X.shape[0], -1)) - np.sort(Z)).mean(axis=1))
plt.title('untrained')
plt.xlabel('dummy experiment')
plt.ylabel('MSE(x, z)')
plt.show()

# %% compare x and reconstruction to debug invertibility problem
plt.plot(np.square(X - X_rec).mean(axis=(1, 2, 3)))
plt.title('untrained')
plt.xlabel('dummy experiment')
plt.ylabel('MSE(x, x\')')
plt.show()

# %%
fig = plt.figure(figsize=(16, 9))#plt.figaspect(0.6))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(np.square(np.sort(X.reshape(X.shape[0], -1)) - np.sort(Z)).mean(axis=1), label='untrained')
#ax3.plot(np.square(np.sort(X.reshape(X.shape[0], -1)) - np.sort(Z_t)).mean(axis=1), label='trained')
#ax3.set(ylabel='MSE(x, z)', xlabel='experiment')
#ax3.set_yscale('log')
#ax1.set_ylim(0)
#ax1.legend()
ax1.set_title('MSE(x, z)')
#ax1.set_ylabel('untrained', rotation=0, size='large')
ax1.annotate('untrained', xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - 0, 0),
                xycoords=ax1.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

ax2.plot(np.square(X - X_rec).mean(axis=(1, 2, 3)), label='untrained')
#ax4.plot(np.square(X - X_rec_t).mean(axis=(1, 2, 3)), label='trained')
#ax4.set(ylabel='MSE(x, x\')', xlabel='experiment')
#ax4.set_yscale('log')
#ax1.set_ylim(0)
#ax2.legend()
ax2.set_title('MSE(x, x\')')

ax3.plot([], label='untrained')
#ax1.plot(np.square(np.sort(X.reshape(X.shape[0], -1)) - np.sort(Z)).mean(axis=1), label='untrained')
ax3.plot(np.square(np.sort(X.reshape(X.shape[0], -1)) - np.sort(Z_t)).mean(axis=1), label='trained')
#ax3.set(ylabel='MSE(x, z)', xlabel='experiment')
ax3.set_xlabel('experiment')
#ax3.set_ylabel('trained', rotation=0, size='large')
ax3.annotate('trained', xy=(0, 0.5), xytext=(-ax3.yaxis.labelpad - 0, 0),
                xycoords=ax3.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
ax3.set_yscale('log')
#ax1.set_ylim(0)
#ax1.legend()

#ax2.plot(np.square(X - X_rec).mean(axis=(1, 2, 3)), label='untrained')
ax4.plot([], label='untrained')
ax4.plot(np.square(X - X_rec_t).mean(axis=(1, 2, 3)), label='trained')
#ax4.set(ylabel='MSE(x, x\')', xlabel='experiment')
ax4.set_xlabel('experiment')
ax4.set_yscale('log')
#ax1.set_ylim(0)
#ax2.legend()

plt.legend()
fig.tight_layout()
plt.show()

# %%
plt.plot([-1, -2, -100, 100])
plt.yscale('log')
plt.show()

# %%

# %%

# %%
# imports
from PIL import Image
import numpy as np
import torch
from models import *

# load net
net = MonetCINN_squeeze(0.01)
net_t = MonetCINN_squeeze(0.01)

new_state_dict = torch.load('checkpoints/monet_cinn.pt')
old_state_dict = net_t.state_dict()

j_old = -1
j_new = -1

for i in range(28, 33):
#print(f"Current permutation index {i}")
    key = f"cinn.module_list.{i}.perm"
    if key in old_state_dict:
        j_old = i
    if key in new_state_dict:
        j_new = i

print(j_old, j_new)
assert(j_old != -1)
assert(j_new != -1)

if j_old != j_new:
    new_state_dict[f"cinn.module_list.{j_old}.perm"]     = new_state_dict[f"cinn.module_list.{j_new}.perm"]
    new_state_dict[f"cinn.module_list.{j_old}.perm_inv"] = new_state_dict[f"cinn.module_list.{j_new}.perm_inv"]
    del new_state_dict[f"cinn.module_list.{j_new}.perm"]
    del new_state_dict[f"cinn.module_list.{j_new}.perm_inv"]

net_t.load_state_dict(new_state_dict)

net.eval()
net_t.eval()

# load data and calc mse
path = 'C:/Users/Jonas/Downloads/cyclegan test/'
N = 100

X_rand = np.zeros((N, 3, 224, 224))
X_img = np.zeros((N, 3, 224, 224))
C_rand = np.zeros((N, 3, 226, 226))
C_img = np.zeros((N, 3, 226, 226))

Z_rand_ut = np.zeros((N, 150528))
Z_rand_t = np.zeros((N, 150528))
Z_img_ut = np.zeros((N, 150528))
Z_img_t = np.zeros((N, 150528))

X_rec_rand_ut = np.zeros((N, 3, 224, 224))
X_rec_rand_t = np.zeros((N, 3, 224, 224))
X_rec_img_ut = np.zeros((N, 3, 224, 224))
X_rec_img_t = np.zeros((N, 3, 224, 224))

for i in range(N):
    print(i)
    # create x
    x_rand = np.random.rand(3, 224, 224).astype(np.float32)
    X_rand[i] = x_rand

    x_img = np.array(Image.open(path + 'fake/fake' + str(i+1) + '.jpg'))
    x_img = np.transpose(x_img, (2,0,1)).astype(np.float32)
    x_img = x_img[:, 16:240, 16:240]
    x_img = x_img / 255
    X_img[i] = x_img

    x_rand = torch.tensor(np.expand_dims(x_rand, 0))
    x_img = torch.tensor(np.expand_dims(x_img, 0))
    
    # create c
    c_rand = np.random.rand(3, 226, 226).astype(np.float32)
    C_rand[i] = c_rand

    c_img = np.array(Image.open(path + 'real/real' + str(i+1) + '.jpg'))
    c_img = np.transpose(c_img, (2,0,1)).astype(np.float32)
    c_img = c_img[:, 15:241, 15:241]
    c_img = c_img / 255
    C_img[i] = c_img

    c_rand = torch.tensor(np.expand_dims(c_rand, 0))
    c_img = torch.tensor(np.expand_dims(c_img, 0))

    # untrained net
    z_rand, _ = net.forward(x_rand, c_rand)
    x_rec_rand, _ = net.reverse_sample(z_rand, c_rand)
    Z_rand_ut[i] = z_rand.detach().numpy().squeeze()
    X_rec_rand_ut[i] = x_rec_rand.detach().numpy().squeeze()

    z_img, _ = net.forward(x_img, c_img)
    x_rec_img, _ = net.reverse_sample(z_img, c_img)
    Z_img_ut[i] = z_img.detach().numpy().squeeze()
    X_rec_img_ut[i] = x_rec_img.detach().numpy().squeeze()

    # trained net
    z_rand, _ = net_t.forward(x_rand, c_rand)
    x_rec_rand, _ = net_t.reverse_sample(z_rand, c_rand)
    Z_rand_t[i] = z_rand.detach().numpy().squeeze()
    X_rec_rand_t[i] = x_rec_rand.detach().numpy().squeeze()

    z_img, _ = net_t.forward(x_img, c_img)
    x_rec_img, _ = net_t.reverse_sample(z_img, c_img)
    Z_img_t[i] = z_img.detach().numpy().squeeze()
    X_rec_img_t[i] = x_rec_img.detach().numpy().squeeze()

# %%
# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 9))#plt.figaspect(0.6))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(np.square(np.sort(X_rand.reshape(X_rand.shape[0], -1)) - np.sort(Z_rand_ut)).mean(axis=1), linestyle='dashed', lw=3, label='random')
ax1.plot(np.square(np.sort(X_img.reshape(X_img.shape[0], -1)) - np.sort(Z_img_ut)).mean(axis=1), linestyle='dotted', lw=4, label='test set')

ax2.plot(np.square(X_rand - X_rec_rand_ut).mean(axis=(1, 2, 3)), linestyle='dashed', lw=3, label='random')
ax2.plot(np.square(X_img - X_rec_img_ut).mean(axis=(1, 2, 3)), linestyle='dotted', lw=4, label='test set')

ax3.plot(np.square(np.sort(X_rand.reshape(X_rand.shape[0], -1)) - np.sort(Z_rand_t)).mean(axis=1), linestyle='dashed', lw=3, label='random')
ax3.plot(np.square(np.sort(X_img.reshape(X_img.shape[0], -1)) - np.sort(Z_img_t)).mean(axis=1), linestyle='dotted', lw=4, label='test set')

ax4.plot(np.square(X_rand - X_rec_rand_t).mean(axis=(1, 2, 3)), linestyle='dashed', lw=3, label='random')
ax4.plot(np.square(X_img - X_rec_img_t).mean(axis=(1, 2, 3)), linestyle='dotted', lw=4, label='test set')

ax1.set_title('MSE(x, z)')
ax2.set_title('MSE(x, x\')')

ax1.annotate('untrained', xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - 0, 0),
                xycoords=ax1.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
ax3.annotate('trained', xy=(0, 0.5), xytext=(-ax3.yaxis.labelpad - 0, 0),
                xycoords=ax3.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

ax3.set_xlabel('experiment')
ax4.set_xlabel('experiment')

ax3.set_yscale('log')
ax4.set_yscale('log')

plt.legend()
fig.tight_layout()
plt.show()

# %%
plt.hist(X_rand[0].flatten())
plt.show()

# %%
plt.hist(X_rec_rand_t[0].flatten())
plt.show()

# %%
plt.scatter(list(range(np.product(X_rec_rand_t[0].shape))), X_rec_rand_t[0].flatten(), s=1)
plt.yscale('log')
plt.show()
