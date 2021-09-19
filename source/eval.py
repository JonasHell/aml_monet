from __future__ import division
 
from os import replace
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import torch
from tqdm import tqdm #tqdm is used to show progress bars

from sklearn.decomposition import PCA

import config as c
import models
import data

cinn = models.MonetCINN_squeeze(0)

print("Loading state dict")
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

cinn.load_state_dict(new_state_dict)

cinn.to(c.device)
cinn.eval()

 

def style_transfer_test_set(temp=0.8, postfix=0, img_folder=c.output_image_folder):
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
            #z, j = cinn.forward(image, condition)
            z = temp * torch.randn(condition.shape[0], c.ndim_total).to(c.device)
            recs, j = cinn.reverse_sample(z, condition)
            recs = recs.cpu().numpy()


            for i, im in enumerate(recs):
                im = np.abs(np.transpose(im, (1,2,0)))
                im[im<0] = 0
                im[im>1] = 1
                #print(im.shape)
                cond_im = images[1][i]
                cond_im = np.abs(np.transpose(cond_im, (1,2,0)))
                cond_im = cond_im[1:-1, 1:-1, :]
                im_im = images[0][i]
                im_im = np.abs(np.transpose(im_im, (1,2,0)))
                
                
                #print(cond_im.shape, im_im.shape)
                final_im = np.concatenate((im_im, cond_im, im), axis=1)
                plt.imsave(join(img_folder, '%.6i_%.3i.png' % (counter, postfix)), final_im)
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
                reconstruction = cinn.reverse_sample(z, condition)[0].reshape(B, -1).cpu().numpy()
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

            rec = cinn.reverse_sample(z, cond)[0]

            var.append(np.mean(np.var(rec, axis=0)))

        print(F'Var (of {n} samples)')
        print(np.mean(var))
        print(F'sqrt(Var) (of {n} samples)')
        print(np.sqrt(np.mean(var)))



def interpolation_grid(val_ind=0, VAL_SELECTION = [0, 1], grid_size=5, max_temp=0.9, interp_power=2, img_folder=c.output_image_folder):
    '''
    Make a grid of a 2D latent space interpolation.
    val_ind:        Which image to use (index in current val. set)
    grid_size:      Grid size in each direction
    max_temp:       Maximum temperature to scale to in each direction (note that the corners
                    will have temperature sqrt(2)*max_temp
    interp_power:   Interpolate with (linspace(-lim**p, +lim**p))**(1/p) instead of linear.
                    Because little happens between t = 0.0...0.7, we don't want this to take up the
                    whole grid. p>1 gives more space to the temperatures closer to 1.
    '''
    steps = np.linspace(-(max_temp**interp_power), max_temp**interp_power, grid_size, endpoint=True)
    steps = np.sign(steps) * np.abs(steps)**(1./interp_power)

    #Iterate over Test Set
    test_im = []
    test_cond = []

    shape = None

    with torch.no_grad():
        for images in tqdm(data.test_loader):
            z, j = cinn(images[0], images[0])
            shape = z.shape
            test_im.append(images[0])
            test_cond.append(images[1])

    print("shape of z", shape)
    shape[0] = 1

    test_cond = torch.cat(test_im, dim=0)
    print(test_cond.size())
    test_cond = torch.stack([test_cond[i] for i in VAL_SELECTION], dim=0)
    print(test_cond.size())
    test_cond = torch.cat([test_cond[val_ind:val_ind + 1]]*grid_size**2, dim=0)
    print(test_cond.size())
    
    test_cond = test_cond.to(c.device)


    def interp_z(z0, z1, a0, a1):
        z_out = []
        for z0_i, z1_i in zip(z0, z1):
            z_out.append(a0 * z0_i + a1 * z1_i)
        return z_out

    torch.manual_seed(c.seed)


    z0 = [torch.randn(shape).to(c.device)]
    z1 = [torch.randn(shape).to(c.device)]

    z_grid = []
    for dk in steps:
        for dl in steps:
            z_grid.append(interp_z(z0, z1, dk, dl))

    z_grid = [torch.cat(z_i, dim=0) for z_i in list(map(list, zip(*z_grid)))]

    print("z_grid", z_grid.size())

    with torch.no_grad():
        rec = cinn.reverse_sample(z_grid, test_cond)

    for i,im in enumerate(rec):
        im = np.transpose(im, (1,2,0))
        plt.imsave(join(c.img_folder, '%.6i_%.3i.png' % (val_ind, i)), im)


def pca_scatter(img_folder=c.output_image_folder):
    ''' Perform PCA on latent space and and visualise test dataset projected onto 2D plane in latent space.'''
    image_characteristics = []

    with torch.no_grad():
        for images in tqdm(data.train_loader):
            colour    = images[0].numpy().mean(axis=(2, 3))
            image     = images[0].to(c.device)
            condition = images[1].to(c.device)
            z, log_j = cinn.forward(image, condition)

            nll = torch.mean(z**2, axis=1) / 2 - log_j / c.ndim_total
            image_characteristics.append([nll.cpu().numpy(), z.cpu().numpy(), colour])


    likelihoods         = np.concatenate([C[0] for C in image_characteristics], axis =0 )
    outputs_combined    = np.concatenate([C[1] for C in image_characteristics], axis=0)
    colours             = np.concatenate([C[2] for C in image_characteristics], axis =0 )
    
    pca = PCA(n_components=2)
    pca.fit(outputs_combined)


    outputs_pca = pca.transform(outputs_combined)

    plt.figure(figsize=(9,9))        
    size = 10 + (40 * (likelihoods - np.min(likelihoods)) / (np.max(likelihoods) - np.min(likelihoods)))**2
    plt.scatter(outputs_pca[:, 0], outputs_pca[:, 1], s = size, c = colours)
    
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.xlabel("# PCA vector 1")
    plt.ylabel("# PCA vector 2")
    plt.title("Training set projected into 2D PCA plane")
    plt.savefig(f"{img_folder}/pcascatter.jpg", dpi=200)
    plt.clf()

def latent_space_pca(n_components = 32, img_folder=c.output_image_folder):
    ''' Perform PCA on latent space and interpolate in latent space with principal eigenvectors.'''
    counter = 0
    image_characteristics = []

    with torch.no_grad():
        for images in tqdm(data.train_loader):
            image     = images[0].to(c.device)
            condition = images[1].to(c.device)
            z, log_j = cinn.forward(image, condition)
            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / c.ndim_total
     
            image_characteristics.append([nll.cpu().numpy(), z.cpu().numpy()])


    log_likeli_combined = np.array([C[0] for C in image_characteristics])
    outputs_combined    = np.concatenate([C[1] for C in image_characteristics], axis=0)

    pca = PCA(n_components=n_components)
    pca.fit(outputs_combined)

    batch_counter = 0
    #Iterate over Test Set
    with torch.no_grad():
        for images in tqdm(data.test_loader):
            image     = images[0].to(c.device)
            condition = images[1].to(c.device)

            #Iterate over components
            for i in range(n_components):
                z_vector = pca.components_[i, :]

                #Interpolate in latent space
                for j, t in enumerate(np.linspace(-200, 200, 40)):
                    z = torch.from_numpy(z_vector) * t
                    norm = torch.max(torch.abs(z))
                    if norm != 0 and norm > 1:
                      z /= norm
                    z = z.repeat(condition.shape[0], 1).to(c.device)d
                    recs, jac = cinn.reverse_sample(z, condition)
                    recs = recs.cpu().numpy()

                    
                    img_counter = batch_counter * c.test_batch_size
                    #Iterate over images in batch
                    for k, im in enumerate(recs):
                        im = np.abs(np.transpose(im, (1,2,0)))
                        im[im<0] = 0
                        im[im>1] = 1

                        cond_im = images[1][k]
                        cond_im = np.abs(np.transpose(cond_im, (1,2,0)))
                        cond_im = cond_im[1:-1, 1:-1, :]
                        im_im = images[0][k]
                        im_im = np.abs(np.transpose(im_im, (1,2,0)))
                        
                        
                        #print(cond_im.shape, im_im.shape)
                        final_im = np.concatenate((im_im, cond_im, im), axis=1)

                        plt.imsave(join(img_folder, '%.6i_%.3i_%.3i.png' % (img_counter, i, j)), final_im)
                        img_counter += 1
            batch_counter += 1

#interpolation_grid()
latent_space_pca()
for i in range(8):
    torch.manual_seed(i+111)
    #style_transfer_test_set(postfix=i)