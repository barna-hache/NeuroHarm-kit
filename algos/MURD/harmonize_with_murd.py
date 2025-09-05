"""
Script for preprocessing and harmonizing MRI images using MURD.

Run on CPU / GPU

"""

import os
import argparse
import shutil
import numpy as np
import nibabel as nib
import ants
from PIL import Image
from natsort import natsorted
import tensorflow as tf
from tempfile import mkdtemp
from test_modified import MURD


TEMPLATE_PATH = "/opt/toolkit/algos/MURD/template_mni_256x256x256_1mm.nii.gz"

def extract_central_slices(image, num_slices):
    """
    Extracts the central axial slices from a 3D image.

    Parameters:
    - image: ANTsImage object.
    - num_slices: Number of central slices to extract.

    Returns:
    - ANTsImage containing the central slices.
    """
    array = image.numpy()
    z_dim = array.shape[2]
    start = (z_dim - num_slices) // 2
    end = start + num_slices
    central_slices = array[:, :, start:end]
    return ants.from_numpy(
        central_slices,
        origin=image.origin,
        spacing=image.spacing[:2] + (image.spacing[2] * (z_dim / num_slices),)
    )


def preprocess_mri(input_mri_path, n_axial_slices):
    """
    Preprocesses the MRI image:
    - Reads the input image.
    - Registers it to the MNI template.
    - Extracts central axial slices.
    - Normalizes each slice.
    - Saves 2.5D slices as PNG images.

    Parameters:
    - input_mri_path: Path to the input MRI image.
    - n_axial_slices: Number of axial slices to extract.

    Returns:
    - normalized_slices: 3D numpy array of normalized slices.
    - temp_dir: Temporary directory containing the PNG slices.
    """
    print("Starting preprocessing...")
    temp_dir = mkdtemp()
    n_axial_slices = int(n_axial_slices)

    # Load reference template
    reference_image = ants.image_read(TEMPLATE_PATH)

    # Load and register input image to the reference
    img = ants.image_read(input_mri_path)
    aligned = ants.registration(
        fixed=reference_image,
        moving=img,
        type_of_transform="Rigid"
    )["warpedmovout"]

    # Extract central axial slices
    central_axial_slices = extract_central_slices(aligned, n_axial_slices)

    # Normalize each slice to [0, 1]
    normalized_slices = np.zeros((256, 256, n_axial_slices))
    for i in range(central_axial_slices.shape[2]):
        slice = central_axial_slices[:, :, i]
        max_val = np.max(slice)
        min_val = np.min(slice)
        if max_val - min_val == 0:
            normalized_slices[:, :, i] = np.zeros_like(slice)
        else:
            normalized_slices[:, :, i] = (slice - min_val) / (max_val - min_val)

    # Save 2.5D slices as PNG images
    for i in range(1, normalized_slices.shape[2] - 1):
        img_slice_25D = np.rot90(normalized_slices[:, :, i - 1:i + 2], k=2)
        img_slice_25D = (img_slice_25D * 255).astype(np.uint8)
        img_slice_25D_pil = Image.fromarray(img_slice_25D)
        img_slice_25D_pil.save(f"{temp_dir}/slice_{i}.png")

    print("Preprocessing completed.")
    return normalized_slices, temp_dir


def harmonize_image(preprocessed_data, image_name, temp_dir, output_dir, save_preprocess):
    """
    Harmonizes the preprocessed MRI slices using the MURD model.

    Parameters:
    - preprocessed_data: 3D numpy array of preprocessed slices.
    - image_name: Name of the output image file.
    - temp_dir: Directory containing the PNG slices.
    - output_dir: Directory to save the harmonized image.
    - save_preprocess: Boolean indicating whether to save the preprocessed image.
    """
    print("Starting harmonization...")

    template_img = nib.load(TEMPLATE_PATH)

    if save_preprocess:
        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True) 

        preprocess_output = os.path.join(preprocessed_dir, image_name)

        nii_image = nib.Nifti1Image(preprocessed_data, affine=template_img.affine, header=template_img.header)
        nib.save(nii_image, preprocess_output)
        print(f"Preprocessed image saved at: {preprocess_output}")

    # Prepare list of slice paths
    slice_paths = [os.path.join(temp_dir, name) for name in natsorted(os.listdir(temp_dir))]

    # Harmonization parameters
    img_type = 'T1'  # Options: 'T1' or 'T2'
    test_mode = 'reference_guided'  # Options: 'site_guided' or 'reference_guided'
    reference_type = "Siemens"  # Options: 'GE', 'Philips', 'Siemens'; None for site-guided mode
    L_dir_out = mkdtemp()

    # Run MURD harmonization
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = MURD(sess, img_type, test_mode, slice_paths, L_dir_out, reference_type)
        gan.build_model()
        gan.test()

    # Load harmonized slices
    harmonized_paths = [os.path.join(L_dir_out, name) for name in natsorted(os.listdir(L_dir_out))]
    image_stack = []
    for path in harmonized_paths:
        img = Image.open(path).convert("L")
        image_stack.append(np.rot90(np.array(img), k=2))

    # Stack slices and save as NIfTI
    stacked_images = np.stack(image_stack, axis=-1)
    nifti_img = nib.Nifti1Image(stacked_images, affine=template_img.affine, header=template_img.header)
    output_path = os.path.join(output_dir, image_name)
    nib.save(nifti_img, output_path)

    print(f"Harmonized image saved at: {output_path}")
    return L_dir_out


parser = argparse.ArgumentParser(description="Preprocess and harmonize MRI images using MURD.")
parser.add_argument("input_image", help="Path to the input MRI image.")
parser.add_argument("--output_dir", default="output", help="Directory to save the harmonized image.")
parser.add_argument("--n_axial_slices", type=int, default=200, help="Number of axial slices to harmonize.")
parser.add_argument("--save_preprocess", type=bool, default=False, help="Whether to save the preprocessed image. Default: False.")
args = parser.parse_args()

# Extract image name from path
image_name = os.path.basename(args.input_image)
print(f"Processing image: {image_name}")

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Preprocess and harmonize the image
preprocessed_data, temp_dir = preprocess_mri(args.input_image, args.n_axial_slices)
L_dir_out = harmonize_image(preprocessed_data, image_name, temp_dir, args.output_dir, args.save_preprocess)

# Clean up temporary directory
shutil.rmtree(L_dir_out)
shutil.rmtree(temp_dir)