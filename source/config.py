#################
# Architecture: #
#################
img_width  = 128
img_height = 128

#############################
# Training hyperparameters: #
#############################
N_epochs    = 10
lr          = 1e-3
batch_size  = 128

#Total number of images used for training
N_train     = 512

#We take the first N_test images from the test dataset for validation and the N_val images after that for validation
N_test      = 128
N_val       = 64

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
output_image_folder     = output_root + 'images'

#######################
# Saving checkpoints: #
#######################

#This is where the training algorithm stores the model
model_output = output_root + 'monet_cinn.pt'
#This is where the evaluation algorithm reas the model from
model_path   = model_output