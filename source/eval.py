from os import replace
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm #tqdm is used to show progress bars
import cv2
from sklearn.decomposition import PCA

import config as c
import models
import data


"""
Load trained model and evaluate model with a range of utility functions. 

Remember to:
- Load correct network
- Set correct paths to image output folder
- Uncomment evaluation function that you need
"""

"""
Custom dataset for holding photo + artistic rendering of photo pairs of RGB images
Supports:
-   cropping images to desired size (128 pixels x 128 pixels by default)
-   normalising images
-   data augmentation
-   adding noise to training data

Returns:
-   Tuples of source images with their respective condition images
"""



#Load final architecture 
cinn = models.MonetCINN_squeeze(0)

#Load stored model
print("Loading state dict")
old_state_dict = cinn.state_dict()
new_state_dict = torch.load(c.model_path)

#Fix for random position of last permutation
j_old = -1
j_new = -1

for i in range(28, 34):
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


 
def style_transfer(cond_index, img_folder=c.output_image_folder):
    '''
    Translate the hand picked data set into different styles once.
    Alternatively adapt data.good_data and data.good_loader to train, test or val
    cond_index:    Index of condition image 
    Returns:
    -   Store 6*224 x 224 RGB-images consisting of:
        the style image A, 
        the style image's condition image B, 
        the condition image the style is applied to C, 
        the resulting image D, 
        the difference between A and B over grey background,
        the style applied to a grey surface


    CAREFUL: 
    test_batch_size = 1 is required for this routine to work
    '''

    style_img, style_cond  = data.good_data[cond_index] #pick data.train_data, data.test_data, data.val_data with appropriate indices when needed
    style_img  = style_img.reshape(1, 3, 224, 224).to(c.device)
    style_cond = style_cond.reshape(1, 3, 226, 226).to(c.device)

    z_style, j_style = cinn.forward(style_img, style_cond)
    z_style          = z_style.repeat(c.test_batch_size, 1)

    style_numpy = np.transpose(style_img[0].cpu().numpy(), (1,2,0))

    grey = np.ones((c.test_batch_size, 3, 226, 226), dtype=np.double) * 128/255
    grey_img = torch.tensor(grey).to(c.device, dtype=torch.float)
    single_grey = np.ones((226, 226, 3))*128/255

    counter = 0
    with torch.no_grad():
        for images in tqdm(data.good_loader):
            condition = images[1].to(c.device)

            rec_grey, j = cinn.reverse_sample(z_style, grey_img)
            rec_grey = rec_grey.cpu().numpy()

            recs, j = cinn.reverse_sample(z_style, condition)
            recs = recs.cpu().numpy()

            grey_im = rec_grey[0]
            grey_im = np.abs(np.transpose(grey_im, (1,2,0)))
            grey_im[grey_im<0] = 0
            grey_im[grey_im>1] = 1

            im = recs[0]
            im = np.abs(np.transpose(im, (1,2,0)))
            im[im<0] = 0
            im[im>1] = 1
            
            cond_im = images[1][0]
            cond_im = np.abs(np.transpose(cond_im, (1,2,0)))
            cond_im = cond_im[1:-1, 1:-1, :]

            im_im = images[0][0]
            im_im = np.abs(np.transpose(im_im, (1,2,0)))

            diff_im = im_im - cond_im + single_grey[1:-1, 1:-1, :]
            diff_im[diff_im<0] = 0
            diff_im[diff_im>1] = 1
            
            final_im = np.concatenate((im_im, cond_im, im, diff_im, grey_im), axis=1)
            plt.imsave(join(img_folder, '%.6i_%.3i.png' % (counter, cond_index)), final_im)
            counter += 1

def style_transfer_test_set(temp=0.8, postfix=0, img_folder=c.output_image_folder):
    '''
    Translate the whole test set into different styles once.
    By uncommenting the cv2.bilateralFilter, resulting images are smoothened
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
                #Enable filtering for smoother image results
                #im = cv2.bilateralFilter(im, 4, 15, 15)
                cond_im = images[1][i]
                cond_im = np.abs(np.transpose(cond_im, (1,2,0)))
                cond_im = cond_im[1:-1, 1:-1, :]
                im_im = images[0][i]
                im_im = np.abs(np.transpose(im_im, (1,2,0)))
                
                
                #print(cond_im.shape, im_im.shape)
                final_im = np.concatenate((im_im, cond_im, im), axis=1)
                plt.imsave(join(img_folder, '%.6i_%.3i.png' % (counter, postfix)), final_im)
                counter += 1


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

def latent_space_pca(n_components = 32, n_sample = 40, img_folder=c.output_image_folder):
    ''' 
    Perform PCA on latent space and interpolate in latent space with the first n_components principal eigenvectors.
    
    The algorithm samples all images in the test set at t*v_i for all i in 1...n_components and for n_sample t in -200 to 200 where v_i is the i-th PCA vector
    These are then stored in the image output folder for the easy creation of GIF animations. 
    By uncommenting the cv2.bilateralFilter, resulting images are smoothened
    '''
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
                for j, t in enumerate(np.linspace(-200, 200, n_sample)):
                    z = torch.from_numpy(z_vector) * t
                    norm = torch.max(torch.abs(z))
                    if norm != 0 and norm > 1:
                      z /= norm
                    z = z.repeat(condition.shape[0], 1).to(c.device)
                    recs, jac = cinn.reverse_sample(z, condition)
                    recs = recs.cpu().numpy()

                    
                    img_counter = batch_counter * c.test_batch_size
                    #Iterate over images in batch
                    for k, im in enumerate(recs):
                        im = np.abs(np.transpose(im, (1,2,0)))
                        im[im<0] = 0
                        im[im>1] = 1
                        #im = cv2.bilateralFilter(im, 3, 20, 20)

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


def latent_space_animation(n_components = 8, n_sample = 10, img_folder=c.output_image_folder):
    ''' 
    Perform PCA on latent space and interpolate in latent space with the first n_components principal eigenvectors.
    
    The algorithm samples all images in the test set and projects their latent codes onto the first n_components PCA eigenvectors. 
    at z = sum_i^n c_i * v_i where c_i are the components of the latent code z of a given image projected onto the PCA eigenvectors v_i 
    The algorithm starts at z = 0, samples ten points from z = 0 to z = v_1, then samples ten points from z = c_1 + v_1 to z = c_1 * v_1 + c_2 * v_2 and so on until we reach
    These are then stored in the image output folder for the easy creation of GIF animations. 

    CAREFUL: 
    This routine creates #N_test * n_components * n_sample images, choose N_test to be around 2 or 3 at maximum. 
    Choose test_batch_size to be 1. 
    '''
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

            z, log_j = cinn.forward(image, condition)
            
            components = pca.transform(z.cpu().numpy())
            print(components.shape)

            #Iterate over components
            for i in range(n_components):
                z_vector = pca.components_[i, :]

                z_orig = np.zeros(pca.components_[i,:].shape)

                for j in range(i):
                  z_orig += components[0, j] * pca.components_[i, :]

                #Interpolate in latent space
                for j, t in enumerate(np.linspace(0, components[0, i], n_sample)):
                    z = z_orig + z_vector * t
                    z = torch.from_numpy(z)
                    z = z.repeat(condition.shape[0], 1).to(c.device)
                    recs, jac = cinn.reverse_sample(z, condition)
                    recs = recs.cpu().numpy()

                    
                    img_counter = batch_counter * c.test_batch_size
                    #Iterate over images in batch
                    for k, im in enumerate(recs):
                        im = np.abs(np.transpose(im, (1,2,0)))
                        im[im<0] = 0
                        im[im>1] = 1
                        im = cv2.bilateralFilter(im, 3, 20, 20)

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

def latent_space_reconstruction(n_components = 1024, img_folder=c.output_image_folder):
    ''' 
    Perform PCA on latent space of training set. 
    
    Store explained variances and variance ratios of PCA vectors in image folder. 
    Project first validation image into PCA space and check how well the first N PCA vectors can approximate the validation image's real latent code. 
    '''

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

    img  = data.val_img_all[0].reshape(1, 3, 224, 224)
    cond = data.val_cond_all[0].reshape(1, 3, 226, 226)

    with torch.no_grad():
      z, log_j = cinn.forward(img, cond)

    z_numpy = z.cpu().numpy()
    components = pca.transform(z_numpy)

    np.savetxt(img_folder + "pca_variance.txt", pca.explained_variance_)
    np.savetxt(img_folder + "pca_variance_ratio.txt", pca.explained_variance_ratio_)
    np.savetxt(img_folder + "pca_eigenvalues.txt", components)

    rec_error = np.zeros(n_components)
    z_rec = np.zeros(z.cpu().numpy().shape)
    for i in range(n_components):
        z_rec += components[0, i] * pca.components_[i, :]
        rec_error[i] = np.linalg.norm(z_numpy - z_rec)

    np.savetxt(img_folder + "pca_reconstruction_quality.txt", rec_error)

        
#latent_space_pca()
#for i in range(4):
    #torch.manual_seed(i+111)
    #style_transfer(i)
    #style_transfer_test_set(temp=0, postfix=i, img_folder=c.output_image_folder+"z0/")
    #style_transfer_test_set(temp=0.5, postfix=i, img_folder=c.output_image_folder+"z5/")
    #style_transfer_test_set(temp=1.0, postfix=i, img_folder=c.output_image_folder+"z10/")