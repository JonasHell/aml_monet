import numpy as np
from skimage import io, color
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T

import cv2
import albumentations

"""
Custom dataset for holding photo + artistic rendering of photo pairs of RGB images
Supports:
-   rescaling images to desired size (128 pixels x 128 pixels by default)
-   normalising images
-   data augmentation
-   adding noise to training data

Returns:
-   Tuples of source images with their respective condition images
"""

class PairDataset(Dataset):

  #Resize dataset, normalise data and augment if is_valid = 0
  def __init__(self, image_paths, condition_paths,  transform = True, noise = False, img_height = 128, img_width = 128, mean = (0, 0, 0), std  = (1, 1, 1)):
    self.image_paths        = image_paths
    self.condition_paths    = condition_paths
    #Check whether there is the same number of images and condition
    assert(len(image_paths) == len(condition_paths))
    self.img_height         = img_height
    self.img_width          = img_width
    self.transform          = transform
    self.noise              = noise


    if self.transform == False:
      self.aug = albumentations.Compose([
                                         albumentations.Resize(img_height, img_width, always_apply = True),
                                         albumentations.Normalize(mean, std, always_apply = True)
      ])
    else:
      #Apply affine transformations to scale, shift and rotate input images
      self.aug = albumentations.Compose([
                                         albumentations.Resize(img_height, img_width, always_apply = True),
                                         albumentations.Normalize(mean, std, always_apply = True),
                                         albumentations.ShiftScaleRotate(shift_limit = 0.0625, 
                                                                         scale_limit = 0.1, 
                                                                         rotate_limit = 5,
                                                                         p = 0.9)
      ])

  def __len__(self):
    return len(self.image_paths)

  def get_image(self, filename):
    #Open image and convert to numpy array 
    img = np.array(Image.open(filename))
    
    #Resize to desired size and interpolate if necessary
    #Normalise and apply data augmentation
    #img = cv2.resize(img, dsize=(self.img_width, self.img_height), interpolation = cv2.INTER_CUBIC)
    img = self.aug(image = img)["image"]

    #Convert back to pytorch data structure
    img = np.transpose(img, (2,0,1)).astype(np.float32) # changing format s.t. pytorch will accept it
    img = torch.tensor(img, dtype = torch.float)

    #Add noise to improve training performance
    if self.noise:
        img += 0.005 * torch.rand_like(img)

    return img

  #Return tuple of image and its condition (another image)
  def __getitem__(self, index):
    image = self.get_image(self.image_paths[index])
    condition = self.get_image(self.condition_paths[index])
    return image, condition


batch_size  = 128
N_train     = 512
N_test      = 128
N_val       = 64

training_img_list   =  [f'/content/gdrive/MyDrive/Daniel-Daten/cyclegan train/fake/fake{i}.jpg' for i in range(1, 1 + N_train)]
training_cond_list  =  [f'/content/gdrive/MyDrive/Daniel-Daten/cyclegan train/real/real{i}.jpg' for i in range(1, 1 + N_train)]
test_img_list       =  [f'/content/gdrive/MyDrive/Daniel-Daten/cyclegan test/fake/fake{i}.jpg' for i in range(1, 1 + N_test)]
test_cond_list      =  [f'/content/gdrive/MyDrive/Daniel-Daten/cyclegan test/real/real{i}.jpg' for i in range(1, 1 + N_test)]
val_img_list        =  [f'/content/gdrive/MyDrive/Daniel-Daten/cyclegan test/fake/fake{i}.jpg' for i in range(2 + N_test, 2 + N_test + N_val)]
val_cond_list       =  [f'/content/gdrive/MyDrive/Daniel-Daten/cyclegan test/real/real{i}.jpg' for i in range(2 + N_test, 2 + N_test + N_val)]

train_data = PairDataset(training_img_list, training_cond_list, transform=True, noise=True)
test_data  = PairDataset(test_img_list, test_cond_list        , transform=False, noise=False)
val_data  =  PairDataset(val_img_list, val_cond_list          , transform=False, noise=False)


train_loader = DataLoader(train_data,   batch_size=batch_size, shuffle=True,    num_workers=8,  pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_data,    batch_size=batch_size, shuffle=False,   num_workers=4,  pin_memory=True, drop_last=False)

"""
Mit den folgenden Zeilen kann man den Code testen und sich den Effekt der Data Augmentation für das Trainingsdatenset anschauen

import matplotlib.pyplot as plt
%matplotlib inline
id = 64
img1, img2 = test_data[id]
plt.imshow(np.transpose(img1.numpy(), (1, 2, 0)))
plt.imshow(np.transpose(img2.numpy(), (1, 2, 0)))


"""