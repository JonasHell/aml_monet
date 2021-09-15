# %%
from time import time

#from tqdm import tqdm
import torch
import torch.optim
import numpy as np

import config as c
import models
import data

"""
Training routine for INN
Remember to:
- Adapt number of epochs
- Set learning rate
- Set output directory for model
"""

cinn = models.MonetCINN_112_blocks10(c.lr)
cinn.cuda()
scheduler = torch.optim.lr_scheduler.StepLR(cinn.optimizer, 1, gamma=0.1)

N_epochs = c.N_epochs
t_start = time()
nll_mean = []

print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
for epoch in range(N_epochs):
    for i, images in enumerate(data.train_loader):
        #train_loader returns a list with two elements
        #Both elements are torch tensors with size (batch_size x 3 (RGB channels) x image width x image height)
        #We immediately load every batch of source and condition images to the GPU
        source    = images[0].cuda()
        condition = images[1].cuda()

        #We pass both the source image as well as the condition image to the INN
        """
        This needs to be adapted depending on the final architecture of the INN
        """
        z, log_j = cinn(source, condition)

        #Compute the loss of the INN
        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / c.ndim_total
        nll.backward()
        nll_mean.append(nll.item())
        cinn.optimizer.step()
        cinn.optimizer.zero_grad()

        #Print training progress every 20 batches
        if not i % 20:
            #Compute training loss for validation images
            with torch.no_grad():
                """
                This needs to be adapted depending on the final architecture of the INN
                """
                z, log_j = cinn(data.val_img_all, data.val_cond_all)
                nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / c.ndim_total

            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                            i, len(data.train_loader),
                                                            (time() - t_start)/60.,
                                                            np.mean(nll_mean),
                                                            nll_val.item(),
                                                            cinn.optimizer.param_groups[0]['lr'],
                                                            ), flush=True)
            nll_mean = []

    scheduler.step()
torch.save(cinn.state_dict(), c.model_output)

# %%
