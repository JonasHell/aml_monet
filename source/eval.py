from os import replace
from os.path import join

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm #tqdm is used to show progress bars
from scipy.spatial import distance_matrix


import config as c
import models
import data

cinn = models.MonetCINN_256_blocks10_nosplit(0)
cinn.cuda()
cinn.eval()
state_dict = {k:v for k,v in torch.load(c.model_path).items() if 'tmp_var' not in k}
cinn.load_state_dict(state_dict)

def style_transfer_test_set(temp=1., postfix=0, img_folder=c.image_folder):
    '''
    Translate the whole test set into different styles once.
    temp:       Sampling temperature
    postfix:    Has to be integer. Append to file name (e.g. to make 10 diverse styles of test set)
    '''
    counter = 0
    with torch.no_grad():
        for images in tqdm(data.test_loader):
            condition = images[1].cuda()
            z = temp * torch.randn(condition.shape[0], models.ndim_total).cuda()

            styles = cinn.reverse_sample(z, condition).cpu().numpy()

            for im in styles:
                im = np.transpose(im, (1,2,0))
                plt.imsave(join(img_folder, '%.6i_%.3i.png' % (counter, postfix)), im)
                counter += 1


def best_of_n(n):
    '''
    computes the best-of-n MSE metric
    by comparing the stylised images to the ground truth
    
    '''
    with torch.no_grad():
        errs_batches = []
        for images in tqdm(data.test_loader, disable=True):
            ground_truth    = images[0]
            condition       = images[1].cuda()

            B = ground_truth.shape[0] # batch_size

            ground_truth = ground_truth.reshape(B, -1) #Flatten RGB ground truth to B x 3*width*height tensor
            errs = np.inf * np.ones(B) #MSE error for each image in batch

            for k in range(n):
                #Randomly sample latent space
                z = torch.randn(B, models.ndim_total).cuda() 
                #Create image using condition and random latent space sample
                reconstruction = cinn.reverse_sample(z, condition).reshape(B, -1).cpu().numpy()
                #Compute vector with length batch_size to quantify how well sampling did compared to ground truth
                errs_k = np.mean((ground_truth - reconstruction)**2, axis=1)
                errs = np.minimum(errs, errs_k)

            errs_batches.append(np.mean(errs))

        print(F'MSE best of {n}')
        print(np.sqrt(np.mean(errs_batches)))
        return np.sqrt(np.mean(errs_batches))

def rgb_var(n):
    '''computes the pixel-wise variance of samples'''
    with torch.no_grad():
        var = []
        for images in tqdm(data.test_all, disable=True):
            cond = images[1].cuda()
            z = torch.randn(n, models.ndim_total).cuda()

            rec = cinn.reverse_sample(z, cond)

            var.append(np.mean(np.var(rec, axis=0)))

        print(F'Var (of {n} samples)')
        print(np.mean(var))
        print(F'sqrt(Var) (of {n} samples)')
        print(np.sqrt(np.mean(var)))

for i in range(8):
    torch.manual_seed(i+111)
    style_transfer_test_set(postfix=i)