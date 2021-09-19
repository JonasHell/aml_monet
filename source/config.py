import torch

#################
# Architecture: #
#################
img_size  = 224
cond_size = 226

ndim_total = 3 * img_size * img_size

#############################
# Training hyperparameters: #
#############################
N_epochs    = 30
lr          = 5e-5
batch_size  = 32
test_batch_size  = 1

#Total number of images used for training
N_train     = 6200

#We take the first N_test images from the test dataset for validation and the N_val images after that for validation
N_test      = 4
N_val       = 64

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
weight_decay = 1e-5
betas = (0.9, 0.999)                # concerning adam optimizer

#######################
# Dataset parameters: #
#######################

#Root folder for dataset
data_root = '/content/gdrive/MyDrive/Daniel-Daten/'

#The code expects the images in the training and test folders to be named real1.jpg, real2.jpg, ..., fake1.jpg, fake2.jpg, ...
training_img_folder    =  data_root + 'cyclegan train/fake/'
training_cond_folder   =  data_root + 'cyclegan train/real/'
test_img_folder        =  data_root + 'cyclegan test/fake/' 
test_cond_folder       =  data_root + 'cyclegan test/real/'

########################
# Display and logging: #
########################

output_root = '/content/gdrive/MyDrive/CINN/'
output_image_folder     = output_root + 'images/'

seed = 314159
#######################
# Saving checkpoints: #
#######################
checkpoint_save_interval = 20   #in epochs
checkpoint_save_overwrite = True #Checkpoints are overwritten if set to True

squeeze_path            = output_root + 'sqeeze/'
vgg11_path              = output_root + 'vgg11/'
output_model_folder     = output_root + 'model/'

#Continue training with model from model_path if set to True
continue_training = False

#This is where the training algorithm stores the model
model_output = output_model_folder + 'monet_cinn.pt'
#This is where the evaluation algorithm reas the model from
model_path   = output_model_folder + 'monet_cinn.pt_cinn_checkpoint_0000'