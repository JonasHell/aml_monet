import numpy as np
from skimage import io, color
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import albumentations

import config as c

"""
Implement dataset for pytorch dataloader that returns source images as well as condition images
and create training and test dataloaders.

Remember to:
- Set correct paths to images
- Adapt image size and data augmentation to your needs
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

class PairDataset(Dataset):

  #Resize dataset, normalise data and augment if is_valid = 0
  def __init__(self, image_paths, condition_paths,  transform = True, noise = False, img_size = c.img_size, cond_size = c.cond_size, mean = (0, 0, 0), std  = (1, 1, 1)):
    self.image_paths        = image_paths
    self.condition_paths    = condition_paths
    #Check whether there is the same number of images and condition
    assert(len(image_paths) == len(condition_paths))
    self.img_size           = img_size
    self.cond_size          = cond_size
    self.transform          = transform
    self.noise              = noise


    if self.transform == False:
      self.img_aug = albumentations.Compose([
                                         albumentations.Resize(img_size, img_size, always_apply = True),
                                         albumentations.Normalize(mean, std, always_apply = True)
      ])
      self.cond_aug = albumentations.Compose([
                                         albumentations.Resize(cond_size, cond_size, always_apply = True),
                                         albumentations.Normalize(mean, std, always_apply = True)
      ])
    else:
      #Apply affine transformations to scale, shift and rotate input images
      self.img_aug = albumentations.Compose([
                                         albumentations.Resize(img_size, img_size, always_apply = True),
                                         albumentations.Normalize(mean, std, always_apply = True),
                                         albumentations.ShiftScaleRotate(shift_limit = 0.0625, 
                                                                         scale_limit = 0.1, 
                                                                         rotate_limit = 5,
                                                                         p = 0.9)
      ])
      #Apply affine transformations to scale, shift and rotate input images
      self.cond_aug = albumentations.Compose([
                                         albumentations.Resize(cond_size, cond_size, always_apply = True),
                                         albumentations.Normalize(mean, std, always_apply = True),
                                         albumentations.ShiftScaleRotate(shift_limit = 0.0625, 
                                                                         scale_limit = 0.1, 
                                                                         rotate_limit = 5,
                                                                         p = 0.9)
      ])

  def __len__(self):
    return len(self.image_paths)

  #Return tuple of image and its condition (another image)
  def __getitem__(self, index):
    #Open image and convert to numpy array 
    image = np.array(Image.open(self.image_paths[index]))
    image = self.img_aug(image = image)["image"]
    image = np.transpose(image, (2,0,1)).astype(np.float32)
    image = torch.tensor(image, dtype = torch.float)
    if self.noise:
        image += 0.005 * torch.rand_like(image)

    #Open image and convert to numpy array 
    condition = np.array(Image.open(self.condition_paths[index]))
    condition = self.cond_aug(image = condition)["image"]
    condition = np.transpose(condition, (2,0,1)).astype(np.float32)
    condition = torch.tensor(condition, dtype = torch.float)
    if self.noise:
        condition += 0.005 * torch.rand_like(condition)

    return image, condition



training_img_list   =  [c.training_img_folder  + f'fake{i}.jpg' for i in range(1, 1 + c.N_train)]
training_cond_list  =  [c.training_cond_folder + f'real{i}.jpg' for i in range(1, 1 + c.N_train)]
test_img_list       =  [c.test_img_folder      + f'fake{i}.jpg' for i in range(1, 1 + c.N_test)]
test_cond_list      =  [c.test_cond_folder     + f'real{i}.jpg' for i in range(1, 1 + c.N_test)]
val_img_list        =  [c.test_img_folder      + f'fake{i}.jpg' for i in range(2 + c.N_test, 2 + c.N_test + c.N_val)]
val_cond_list       =  [c.test_cond_folder     + f'real{i}.jpg' for i in range(2 + c.N_test, 2 + c.N_test + c.N_val)]

train_data = PairDataset(training_img_list, training_cond_list, transform=True, noise=True)
test_data  = PairDataset(test_img_list, test_cond_list        , transform=False, noise=False)
val_data  =  PairDataset(val_img_list, val_cond_list          , transform=False, noise=False)


train_loader = DataLoader(train_data,   batch_size=c.batch_size, shuffle=True,    num_workers=2,  pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_data,    batch_size=c.batch_size, shuffle=False,   num_workers=2,  pin_memory=True, drop_last=False)

#Load all test and validation images and append them to a list
#stack concatenates a sequence of tensors along a new dimension
#list creates a list using the __get_item__ function
x  = list(test_data)
tx = list(zip(*x))
test_img_all  = torch.stack(tx[0], 0).to(c.device)
test_cond_all  = torch.stack(tx[1], 0).to(c.device)

x  = list(val_data)
tx = list(zip(*x))
val_img_all  = torch.stack(tx[0], 0).to(c.device)
val_cond_all  = torch.stack(tx[1], 0).to(c.device)
