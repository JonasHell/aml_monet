from os import replace
from os.path import join

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm #tqdm is used to show progress bars
from scipy.spatial import distance_matrix

from sklearn.decomposition import PCA

import config as c
import models
import data

cinn = models.MonetCINN_112_blocks10(0)
cinn.to(c.device)
state_dict = torch.load(c.model_path)
for key in state_dict.keys():
  print(key)
state_dict["cinn.module_list.28.perm"]    = state_dict["cinn.module_list.32.perm"]
state_dict["cinn.module_list.29.perm"]    = state_dict["cinn.module_list.32.perm"]
state_dict["cinn.module_list.30.perm"]    = state_dict["cinn.module_list.32.perm"]
state_dict["cinn.module_list.31.perm"]    = state_dict["cinn.module_list.32.perm"]
state_dict["cinn.module_list.28.perm_inv"] = state_dict["cinn.module_list.32.perm_inv"]
state_dict["cinn.module_list.29.perm_inv"] = state_dict["cinn.module_list.32.perm_inv"]
state_dict["cinn.module_list.30.perm_inv"] = state_dict["cinn.module_list.32.perm_inv"]
state_dict["cinn.module_list.31.perm_inv"] = state_dict["cinn.module_list.32.perm_inv"]
#state_dict = {k:v for k,v in torch.load(c.model_path).items() if 'tmp_var' not in k}
cinn.load_state_dict(state_dict, strict = False)

cinn.eval()

def style_transfer_test_set(temp=1., postfix=0, img_folder=c.output_image_folder):
    '''
    Translate the whole test set into different styles once.
    temp:       Sampling temperature
    postfix:    Has to be integer. Append to file name (e.g. to make 10 diverse styles of test set)
    '''
    counter = 0
    with torch.no_grad():
        for images in tqdm(data.test_loader):
            image     = images[0].to(c.device)
            condition = images[1].to(c.device)
            #z = temp * torch.randn(condition.shape[0], c.ndim_total).to(c.device)
            z, j = cinn.forward(image, condition)
            z += 0.002*temp * torch.randn(condition.shape[0], c.ndim_total).to(c.device)
            recs, j = cinn.reverse_sample(z, condition)
            recs = recs.cpu().numpy()

            for im in recs:
                im = np.abs(np.transpose(im, (1,2,0)))
                im[im<0] = 0
                im[im>1] = 1
                #print(im)
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
            condition       = images[1].to(c.device)

            B = ground_truth.shape[0] # batch_size

            ground_truth = ground_truth.reshape(B, -1) #Flatten RGB ground truth to B x 3*width*height tensor
            errs = np.inf * np.ones(B) #MSE error for each image in batch

            for k in range(n):
                #Randomly sample latent space
                z = torch.randn(B, models.ndim_total).to(c.device) 
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
            cond = images[1].to(c.device)
            z = torch.randn(n, models.ndim_total).to(c.device)

            rec = cinn.reverse_sample(z, cond)

            var.append(np.mean(np.var(rec, axis=0)))

        print(F'Var (of {n} samples)')
        print(np.mean(var))
        print(F'sqrt(Var) (of {n} samples)')
        print(np.sqrt(np.mean(var)))

def latent_space_pca(n_components = 2, img_folder=c.output_image_folder):
    ''' Perform PCA on latent space to see where images lie in relation to each other.'''
    counter = 0
    image_characteristics = []

    with torch.no_grad():
        for images in tqdm(data.test_loader):
            image     = images[0].to(c.device)
            condition = images[1].to(c.device)
            z, log_j = cinn.forward(image, condition)

            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / c.ndim_total
            z = z.cpu().numpy()

            image_characteristics.append([nll, z])


    log_likeli_combined = np.concatenate([C[0] for C in image_characteristics], axis=0)
    outputs_combined    = np.concatenate([C[1] for C in image_characteristics], axis=0)

    pca = PCA(n_components=n_components)
    pca.fit(outputs_combined)

    with torch.no_grad():
        for images in tqdm(data.test_loader):
            for i in range(n_components):
                z_vector = pca.components_[i, :]
                image     = images[0].to(c.device)

                for t in np.linspace(-1, -1, 10):
                    recs, j = cinn.reverse_sample(z_vector*t, condition)
                    recs = recs.cpu().numpy()

                    for im in recs:
                        im = np.abs(np.transpose(im, (1,2,0)))
                        im[im<0] = 0
                        im[im>1] = 1
                        #print(im)
                        plt.imsave(f"{img_folder}, {counter}, {t}.jpg", im)
                        counter += 1


for i in range(8):
    torch.manual_seed(i+111)
    style_transfer_test_set(postfix=i)