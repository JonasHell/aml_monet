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

cinn = models.MonetCINN_squeeze(c.lr)

loss = 0
last_epoch = 0

print("Loading state dict")
if c.continue_training:
  old_state_dict = cinn.state_dict()
  new_state_dict = torch.load(c.model_path)

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

  #state_dict = {k:v for k,v in torch.load(c.model_path).items() if 'tmp_var' not in k}
  cinn.load_state_dict(new_state_dict)
  
  with open("loss.txt", "r") as file:
      for last_line in file:
          pass
  last_epoch = int(float(last_line.split("\t")[0])) + 1
else:
  loss = open("loss.txt", "w")
  loss.write('Epoch\tTime\tLoss train\tLoss val\n')
  loss.close()

cinn.to(c.device)
scheduler = torch.optim.lr_scheduler.StepLR(cinn.optimizer, 10, gamma=0.1)

N_epochs = c.N_epochs
t_start = time()
nll_mean = np.array([])

print('Epoch\tBatch/Total \tTime \tLoss train \tLoss val \t1st term \t2nd term \tLR')
for epoch in range(N_epochs):
    loss = open("loss.txt", "a")
    for i, images in enumerate(data.train_loader):
        #train_loader returns a list with two elements
        #Both elements are torch tensors with size (batch_size x 3 (RGB channels) x image width x image height)
        #We immediately load every batch of source and condition images to the GPU
        source    = images[0].to(c.device)
        condition = images[1].to(c.device)

        #We pass both the source image as well as the condition image to the INN
        """
        This needs to be adapted depending on the final architecture of the INN
        """
        z, log_j = cinn(source, condition)

        #Compute the loss of the INN, z is batch size x 3*img_size**2 array whereas log_j is batch size array
        #This is why we need to divide torch.mean(log_j) by 3*img_size**2 separately
        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / c.ndim_total
        nll.backward()
        if epoch > 0: nll_mean = np.delete(nll_mean, 0) #delete oldest entry from nll_mean such that the mean is for 1 epoch
        nll_mean = np.append(nll_mean, nll.item())
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
                nll1 = torch.mean(z**2) / 2
                nll2 = torch.mean(log_j) / c.ndim_total
                nll_val = nll1 - nll2
            loss.write('%.2f\t%.2f\t%.6f\t%.6f\n' % (last_epoch + epoch + i/len(data.train_loader), (time() - t_start)/60., np.mean(nll_mean), nll_val.item()))
            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.6f\t%.6f\t%.2e' % (epoch, i, len(data.train_loader), (time() - t_start)/60., np.mean(nll_mean), nll_val.item(), nll1.item(), -nll2.item(), cinn.optimizer.param_groups[0]['lr'], ), flush=True)
    
    loss.close()
    if epoch > 0 and (epoch % c.checkpoint_save_interval) == 0:
        torch.save(cinn.state_dict(), c.model_output + '_cinn_checkpoint_%.4i' % (epoch * (1-int(c.checkpoint_save_overwrite == True))))

    scheduler.step()
torch.save(cinn.state_dict(), c.model_output)

# %%
