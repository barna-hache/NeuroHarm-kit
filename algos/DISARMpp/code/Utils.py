import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib.colors as mcolors
import torch.nn as nn
from tqdm import tqdm
from Load_Dataset import *
from tensorboardX import SummaryWriter


def get_axis_slices(img, slice_index):
        num_rows = 1
        num_cols = 3
        names = ['Sagittal', 'Axial', 'Coronal']
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
     
        for j in range(num_cols):
                if(j == 0):
                    z_slice = img[:, :, slice_index]
                if(j == 1):
                    z_slice = img[slice_index, :, :]
                if(j == 2):
                    z_slice = img[:, (slice_index+18), :]
                # Plot the 2D slice in the current axis
                axes[j].imshow(z_slice, cmap='gray')
                axes[j].set_title(f'T1 - {names[j]}', fontsize=20)
        plt.tight_layout()
        plt.show()


def get_axis_slices_90_rot(img, slice_index):
        num_rows = 1
        num_cols = 3
        names = [ 'Sagittal','Coronal','Axial' ]
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
     
        for j in range(num_cols):
                if(j == 0):
                    z_slice = np.rot90(img[slice_index, :, :])
                if(j == 1):
                    z_slice = np.rot90(img[:, (slice_index+18), :])
                if(j == 2):
                    z_slice = np.rot90(img[:, :, slice_index])
                # Plot the 2D slice in the current axis
                axes[j].imshow(z_slice, cmap='gray')
                axes[j].set_title(f'T1 - {names[j]}', fontsize=20)
        plt.tight_layout()
        plt.show()

def tensor2img(img):
    img = img.cpu().float().detach().numpy()
    img = (img + 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def nib_tensor2img(img):
    img = img.cpu().float().detach().numpy()
    img = (img + 1) / 2.0 * 255.0
    return img

def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler


class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.display_freq = opts.display_freq
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq

    # make directory
    if not os.path.exists(self.display_dir):
        os.makedirs(self.display_dir)
    if not os.path.exists(self.model_dir):
        os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
        os.makedirs(self.image_dir)

    # create tensorboard writer
    self.writer = SummaryWriter(log_dir=self.display_dir)

  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
        assembled_images1, assembled_images2, assembled_images3 = model.assemble_outputs()
        img_filename1 = '%s/gen_%05d_slice_1.jpg' % (self.image_dir, ep)
        img_filename2 = '%s/gen_%05d_slice_2.jpg' % (self.image_dir, ep)
        img_filename3 = '%s/gen_%05d_slice_3.jpg' % (self.image_dir, ep)
        pil_image1 = Image.fromarray(assembled_images1.squeeze(), mode='L')
        pil_image2 = Image.fromarray(assembled_images2.squeeze(), mode='L')
        pil_image3 = Image.fromarray(assembled_images3.squeeze(), mode='L')
        pil_image1.save(img_filename1)
        pil_image2.save(img_filename2)
        pil_image3.save(img_filename3)
    elif ep == -1:
        assembled_images1, assembled_images2, assembled_images3 = model.assemble_outputs()
        img_filename1 = '%s/gen_last_slice_1.jpg' % (self.image_dir, ep)
        img_filename2 = '%s/gen_last_slice_2.jpg' % (self.image_dir, ep)
        img_filename3 = '%s/gen_last_slice_3_.jpg' % (self.image_dir, ep)
        pil_image1 = Image.fromarray(assembled_images1.squeeze(), mode='L')
        pil_image2 = Image.fromarray(assembled_images2.squeeze(), mode='L')
        pil_image3 = Image.fromarray(assembled_images3.squeeze(), mode='L')
        pil_image1.save(img_filename1)
        pil_image2.save(img_filename2)
        pil_image3.save(img_filename3)

  # save model
  def write_model(self, ep, total_it, model):
    if (ep + 1) % self.model_save_freq == 1:
        print('--- save the model @ ep %d ---' % (ep))
        model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
    elif ep == -1:
        model.save('%s/last.pth' % self.model_dir, ep, total_it)


# ### Utils for testing phase

def get_z_random(batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz)
    return z

def transfer_to_scannerfree(source_img, opts, model):
        z_random = get_z_random(source_img.size(0), 8, 'gauss').cuda().float()
        source_img_ = source_img.cuda().float()
        with torch.no_grad():
            output_test = model.test_scannerfree_transfer(source_img_, z_random)
        return output_test[0,0]


def recompose_image_to_reference(image, ref_scanner, opts, model, subset_size=26, moving_window = 1):
    # Ensure the image is a torch tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch tensor.")
    
    # Image dimensions
    d, h, w = image.shape
    
    # Initialize an empty tensor to hold the recomposed image
    recomposed_image = torch.zeros((d, h, w), dtype=image.dtype, device=image.device)
    # Initialize a counter tensor to keep track of the number of additions for averaging
    count = torch.zeros((d, h, w), dtype=torch.float32, device=image.device)
    
    dataset_dom = data_single_std(opts, ref_scanner)
    dataloader_dom = torch.utils.data.DataLoader(dataset_dom, batch_size=1, shuffle=True, num_workers=6)
    for batch in dataloader_dom:
            img, lab = batch

    # Slide the subset across the first dimension
    start = 0
    while start + subset_size <= d:
        end = start + subset_size
        
        # Extract the subset from the original image
        subset_image = image[start:end].unsqueeze(0).unsqueeze(0)
        img_ = img[:, :, start:end, :, :].cuda().float()
        lab_ = lab.cuda().float()
        subset_image_ = subset_image.cuda().float()
        with torch.no_grad():
               gen_subset_image = model.test_reference_transfer(image = subset_image_, image_trg = img_, c_trg = lab_)
                
        # Update the recomposed image and count tensors
        recomposed_image[start:end] += gen_subset_image[0,0]
        count[start:end] += 1
        
        # Move the window one index ahead
        start += moving_window
    
    # Avoid division by zero and compute the average
    # Replace zero counts with one to avoid division errors
    count = torch.where(count == 0, torch.tensor(1.0, device=image.device), count)
    recomposed_image /= count
    
    return recomposed_image


def base_recompose_image_to_reference(image, ref_img, ref_lab, opts, model, subset_size=26, moving_window = 1):
    # Ensure the image is a torch tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch tensor.")
    
    # Image dimensions
    d, h, w = image.shape
    
    # Initialize an empty tensor to hold the recomposed image
    recomposed_image = torch.zeros((d, h, w), dtype=image.dtype, device=image.device)
    # Initialize a counter tensor to keep track of the number of additions for averaging
    count = torch.zeros((d, h, w), dtype=torch.float32, device=image.device)

    # Slide the subset across the first dimension
    start = 0
    while start + subset_size <= d:
        end = start + subset_size
        
        # Extract the subset from the original image
        subset_image = image[start:end].unsqueeze(0).unsqueeze(0)
        img_ = ref_img[:, :, start:end, :, :].cuda().float()
        lab_ = ref_lab.cuda().float()
        subset_image_ = subset_image.cuda().float()
        with torch.no_grad():
               gen_subset_image = model.test_reference_transfer(image = subset_image_, image_trg = img_, c_trg = lab_)
                
        # Update the recomposed image and count tensors
        recomposed_image[start:end] += gen_subset_image[0,0]
        count[start:end] += 1
        
        # Move the window one index ahead
        start += moving_window
    
    # Avoid division by zero and compute the average
    # Replace zero counts with one to avoid division errors
    count = torch.where(count == 0, torch.tensor(1.0, device=image.device), count)
    recomposed_image /= count
    
    return recomposed_image


def custom_transform(x):
    return x.unsqueeze(0)

def custom_permute(x):
    return x.permute(0, 2, 3, 1)

def transfer_img_list_to_reference(img_list, ref_scanner, opts, model):
    list_new = []
    
    transforms = [ToTensor()]
    transforms.append(custom_transform)
    transforms.append(custom_permute)
    transforms.append(tio.RescaleIntensity((-1, 1)))
    transforms = Compose(transforms)
    
    img_list = [transforms(img).unsqueeze(0) for img in img_list]
    dataset_dom = data_single_std(opts, ref_scanner)
    dataloader_dom = torch.utils.data.DataLoader(dataset_dom, batch_size=1, shuffle=True, num_workers=8)
    for batch in dataloader_dom:
            ref_img, ref_lab = batch
    
    for i in tqdm(range(len(img_list)), desc="Harmonizing images to reference scanner"):
        img_ = img_list[i][0,0].cuda().float()
        output = base_recompose_image_to_reference(img_, ref_img, ref_lab, opts, model, subset_size=26, moving_window = 1)
        list_new.append(output)
    return list_new

def transfer_img_list_to_reference2(img_list, ref_img, ref_scanner, opts, model):
    list_new = []
    
    transforms = [ToTensor()]
    transforms.append(custom_transform)
    transforms.append(custom_permute)
    transforms.append(tio.RescaleIntensity((-1, 1)))
    transforms = Compose(transforms)
    
    img_list = [transforms(img).unsqueeze(0) for img in img_list]
    ref_img = transforms(ref_img).unsqueeze(0)
    ref_lab = np.zeros((1, opts.num_domains), dtype=float)
    ref_lab[0, ref_scanner] = 1.0
    
    for i in tqdm(range(len(img_list)), desc="Harmonizing images to reference scanner"):
        img_ = img_list[i][0,0].cuda().float()
        ref_img_ = ref_img.cuda().float()
        ref_lab_ = torch.from_numpy(ref_lab).cuda().float()
        output = base_recompose_image_to_reference(img_, ref_img_, ref_lab_, opts, model, subset_size=26, moving_window = 1)
        list_new.append(output)
    return list_new


def recompose_image_to_scannerfree(image, opts, model,  z_rand, subset_size=26, moving_window = 1):
    # Ensure the image is a torch tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch tensor.")
    
    # Image dimensions
    d, h, w = image.shape
    
    # Initialize an empty tensor to hold the recomposed image
    recomposed_image = torch.zeros((d, h, w), dtype=image.dtype, device=image.device)
    # Initialize a counter tensor to keep track of the number of additions for averaging
    count = torch.zeros((d, h, w), dtype=torch.float32, device=image.device)
    
    z_random = z_rand.cuda().float()

    # Slide the subset across the first dimension
    start = 0
    while start + subset_size <= d:
        end = start + subset_size
        
        # Extract the subset from the original image
        subset_image = image[start:end].unsqueeze(0).unsqueeze(0)
        subset_image_ = subset_image.cuda().float()
                
        with torch.no_grad():
            gen_subset_image = model.test_scannerfree_transfer(subset_image_ , z_random)
                
        # Update the recomposed image and count tensors
        recomposed_image[start:end] += gen_subset_image[0,0]
        count[start:end] += 1
        
        # Move the window one index ahead
        start += moving_window
    
    # Avoid division by zero and compute the average
    # Replace zero counts with one to avoid division errors
    count = torch.where(count == 0, torch.tensor(1.0, device=image.device), count)
    recomposed_image /= count
    
    return recomposed_image


def transfer_img_list_to_scannerfree(img_list, z_rand, opts, model):
    list_new = []
    
    transforms = [ToTensor()]
    transforms.append(custom_transform)
    transforms.append(custom_permute)
    transforms.append(tio.RescaleIntensity((-1, 1)))
    transforms = Compose(transforms)
    
    img_list = [transforms(img).unsqueeze(0) for img in img_list]
    
    for i in tqdm(range(len(img_list)), desc="Harmonizing images to scanner-free"):
        img_ = img_list[i][0,0].cuda().float()
        output = recompose_image_to_scannerfree(img_, opts, model,  z_rand, subset_size=26, moving_window = 1)
        list_new.append(output)
    return list_new


# Function to load .nii.gz images and return both images and filenames
def load_nifti_images_from_folder(folder_path):
    nifti_images = []  # List to store the numpy arrays
    filenames = []  # List to store filenames
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            
            # Load the NIfTI file
            img = nib.load(file_path)
            
            # Convert the NIfTI image to a numpy array
            nifti_images.append(img.get_fdata())
            
            # Append the filename (without the directory path) to the filenames list
            filenames.append(filename)
    
    return nifti_images, filenames

# Function to save the preprocessed images with a prefix to the same filenames
def save_preprocessed_images(folder_path, nifti_images, filenames, temp_dir, prefix="harm_"):
    for img_array, filename in zip(nifti_images, filenames):
        # Create the new filename with the added prefix
        new_filename = prefix + filename

        # Create the full path for the new file
        new_file_path = os.path.join(folder_path, new_filename)
        

        print(f"{temp_dir}/preprocessed/registered_image.nii.gz")
        # Convert the numpy array back to a NIfTI image
        img_affine = nib.load(f"{temp_dir}/preprocessed/registered_image.nii.gz").affine
        new_img = nib.Nifti1Image(img_array, affine=img_affine)  # Use identity matrix for affine
        nib.save(new_img, new_file_path)
        
        print(f"Saved preprocessed image: {new_file_path}")

