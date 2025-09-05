
"""
Script for preprocessing and harmonizing MRI images using STGAN.

Run on GPU only

"""

import numpy as np
import os
import subprocess
import shutil
import argparse
import logging
from tempfile import mkdtemp

import torch
import nibabel as nib
from HD_BET.run import run_hd_bet

# Set environment variable to avoid MKL threading errors
os.environ["MKL_SERVICE_FORCE_INTEL"] = "TRUE"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_cmd(command):
    """
    Execute a system command and log its output in real-time.

    Parameters:
    - command: The system command to execute as a string.

    Raises:
    - subprocess.CalledProcessError: If the command exits with a non-zero status.
    """
    logger.info(f"Executing command: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Stream and log each line of output
    for line in process.stdout:
        logger.info(line.strip())

    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        logger.error(f"Command failed with return code {return_code}")
        raise subprocess.CalledProcessError(return_code, command)


def normalize_image_to_uint8(input_nifti, output_nifti):
    img = nib.load(input_nifti)
    data = img.get_fdata()
    # Normalisation linéaire min-max → [0,255]
    min_val, max_val = np.min(data), np.max(data)
    if max_val > min_val:
        scaled = (data - min_val) / (max_val - min_val) * 255.0
    else:
        scaled = np.zeros_like(data)
    # Conversion en uint8
    scaled_uint8 = scaled.astype(np.uint16)
    # Création du nouveau fichier NIfTI avec les mêmes métadonnées
    new_img = nib.Nifti1Image(scaled_uint8, img.affine, img.header)
    nib.save(new_img, output_nifti)


def preprocess_mri(input_mri_path, apply_preproc_steps):
    """
    Preprocess the input MRI image based on specified steps.

    Args:
        input_mri_path (str): Path to the input MRI image.
        steps (list): List of preprocessing steps to perform.

    Returns:
        tuple: Path to the preprocessed image and the temporary directory used.
    """
    temp_dir = mkdtemp()
    current_input = input_mri_path

    if apply_preproc_steps:
        logger.info("Step 1: Reorienting the image to standard orientation.")
        reorient_output = os.path.join(temp_dir, "reorient.nii.gz")
        reorient_cmd = f"fslreorient2std {current_input} {reorient_output}"
        run_cmd(reorient_cmd)
        current_input = reorient_output

        logger.info("Step 2: Performing brain extraction using HD-BET.")
        # available_gpu = torch.cuda.is_available()
        available_gpu = False
        hd_bet_output = os.path.join(temp_dir, "hd_bet_extraction.nii.gz")
        # device = 'cuda' if available_gpu else 'cpu'
        if available_gpu:
            run_hd_bet(current_input, hd_bet_output, bet=True)
        else:
            run_hd_bet(current_input, hd_bet_output, bet=True, device='cpu', mode='fast', do_tta=False)

        # run_hd_bet(current_input, hd_bet_output, bet=True, device=device, mode='fast', do_tta=False)
        current_input = hd_bet_output

        logger.info("Step 3: Applying N3 bias field correction.")
        n3_output = os.path.join(temp_dir, "n3_corrected.nii.gz")
        # Ajustez les paramètres selon vos besoins
        run_cmd(
            f"N3BiasFieldCorrection 3 {current_input} {n3_output} "
        )

        logger.info("Step 4: Normalizing image intensity to [0, 255].")
        current_input = n3_output
        norm_output = os.path.join(temp_dir, "normalized_uint8.nii.gz")
        normalize_image_to_uint8(current_input, norm_output)
        current_input = norm_output


        logger.info("Step 3: Preprocessing image with STGAN.")
        preprocess_stgan_cmd = (
            f"sh /opt/toolkit/algos/STGAN/pre_process.sh "
            f"{temp_dir}/ "
            f"{temp_dir}/preprocessed "
        )
        run_cmd(preprocess_stgan_cmd)
        current_input = os.path.join(temp_dir, "preprocessed", "normalized_uint8.nii.gz")

        # Clean up intermediate files
        preprocessed_dir = os.path.join(temp_dir, "preprocessed")
        for item in os.listdir(preprocessed_dir):
            item_path = os.path.join(preprocessed_dir, item)
            if os.path.abspath(item_path) != os.path.abspath(current_input):
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    return current_input, temp_dir


def harmonize_image(input_mri, image_name, temp_dir, output_dir, save_preprocess):
    """
    Harmonize the preprocessed MRI image using STGAN.

    Args:
        input_mri (str): Path to the preprocessed MRI image.
        image_name (str): Name of the original image file.
        temp_dir (str): Temporary directory used for processing.
        output_dir (str): Directory to save the harmonized image.
        save_preprocess (bool): Whether to save the preprocessed image.
    """
    if save_preprocess:
        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True) 

        preprocess_output = os.path.join(preprocessed_dir, image_name)

        shutil.copy(input_mri, preprocess_output)
        logger.info(f"Preprocessed image saved at: {preprocess_output}")

    logger.info("Step 4: Harmonizing image with STGAN.")
    harmonize_cmd = (
        f"sh /opt/toolkit/algos/STGAN/harmonize_images.sh "
        f"/opt/toolkit/algos/STGAN/demo/ref/00210_t1_final_mask_ds.nii.gz "
        f"{temp_dir}/preprocessed "
        f"{temp_dir}/harmonized "
        f"/opt/toolkit/algos/STGAN/expr_256 "
    )
    run_cmd(harmonize_cmd)

    harmonized_image_path = os.path.join(
        temp_dir, "harmonized", "normalized_uint8", "normalized_uint8_harm_whead_rescaled_masked.nii.gz"
    )
    destination_path = os.path.join(output_dir, image_name)
    shutil.copy(harmonized_image_path, destination_path)
    logger.info(f"Harmonized image saved at: {destination_path}")


parser = argparse.ArgumentParser(description="Preprocess and harmonize MRI images using STGAN.")
parser.add_argument("input_image", help="Path to the input MRI image.")
parser.add_argument("--apply_preproc_steps", type=bool, default=True,
                    help="Preprocessing steps to perform. Applied preproc steps: brain_extraction, N3, [0-255] intensity normalization, stgan_preprocess. Default: True.")
parser.add_argument("--output_dir", default="output",
                    help="Directory to save the harmonized image.")
parser.add_argument("--save_preprocess", default=False,
                    help="Save preprocessed image.")

args = parser.parse_args()

image_name = os.path.basename(args.input_image)

os.makedirs(args.output_dir, exist_ok=True)

preprocessed_image, temp_dir = preprocess_mri(args.input_image, args.apply_preproc_steps)
harmonize_image(preprocessed_image, image_name, temp_dir, args.output_dir, args.save_preprocess)

shutil.rmtree(temp_dir)
logger.info(f"Temporary directory {temp_dir} removed.")
