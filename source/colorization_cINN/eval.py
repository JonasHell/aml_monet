
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

cinn = models.MonetCINN_VGG(0)
cinn.to(c.device)
state_dict = torch.load(c.model_path)
for key in state_dict.keys():
  print(key)
state_dict["cinn.module_list.28.perm"]    = state_dict["cinn.module_list.30.perm"]
state_dict["cinn.module_list.29.perm"]    = state_dict["cinn.module_list.30.perm"]
state_dict["cinn.module_list.30.perm"]    = state_dict["cinn.module_list.30.perm"]
state_dict["cinn.module_list.31.perm"]    = state_dict["cinn.module_list.30.perm"]
state_dict["cinn.module_list.28.perm_inv"] = state_dict["cinn.module_list.30.perm_inv"]
state_dict["cinn.module_list.29.perm_inv"] = state_dict["cinn.module_list.30.perm_inv"]
state_dict["cinn.module_list.30.perm_inv"] = state_dict["cinn.module_list.30.perm_inv"]
state_dict["cinn.module_list.31.perm_inv"] = state_dict["cinn.module_list.30.perm_inv"]
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

 
class AnimatedGif:
    def __init__(self, size=(c.img_size, c.img_size)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []
 
    def add(self, image, label=''):
        plt_im = plt.imshow(image, vmin=0, vmax=1, animated=True)
        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])
 
    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer="pillow", fps=2)


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

def latent_space_pca(n_components = 2, img_folder=c.output_image_folder):
    ''' Perform PCA on latent space and interpolate in latent space with principal eigenvectors.'''
    counter = 0
    image_characteristics = []

    with torch.no_grad():
        for images in tqdm(data.train_loader):
            image     = images[0].to(c.device)
            condition = images[1].to(c.device)
            z, log_j = cinn.forward(image, condition)
            print(z.shape)

            nll = torch.mean(z**2) / 2 - torch.mean(log_j) / c.ndim_total
     
            image_characteristics.append([nll.cpu().numpy(), z.cpu().numpy()])


    log_likeli_combined = np.array([C[0] for C in image_characteristics])
    outputs_combined    = np.concatenate([C[1] for C in image_characteristics], axis=0)

    pca = PCA(n_components=n_components)
    pca.fit(outputs_combined)

    batch = 0
    #Iterate over Test Set
    with torch.no_grad():
        for images in tqdm(data.test_loader):
            animation_images = []
            condition = images[1].to(c.device)

            #Iterate over components
            for i in range(n_components):
                z_vector = pca.components_[i, :]

                animation_images.append([])
                
                for k in range(condition.shape[0]):
                    animation_images[-1].append([])

                #Interpolate in latent space
                for j, t in enumerate(np.linspace(-200, 200, 20)):
                    z = torch.from_numpy(z_vector) * t
                    z = z.repeat(1, condition.shape[0]).to(c.device)
                    recs, jac = cinn.reverse_sample(z, condition)
                    recs = recs.cpu().numpy()

                    #Iterate over images in batch
                    for k, im in enumerate(recs):
                        im = np.abs(np.transpose(im, (1,2,0)))
                        im[im<0] = 0
                        im[im>1] = 1
                        animation_images[-1][k].append(im)
            #Store images in batch as gif animation
            for i, components in enumerate(animation_images):
                for j, photo in enumerate(components):
                    gif = AnimatedGif()
                    for k, frame in enumerate(photo):
                        gif.add(frame, label=f"Frame: {k}")
                    gif.save(filename= f"{img_folder}/img_{j}_batch_{batch}_component_{i}.gif")
            batch += 1

pca_scatter()

for i in range(8):
    torch.manual_seed(i+111)
    #style_transfer_test_set(postfix=i)