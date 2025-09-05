import os
import subprocess
import shutil
import argparse
import logging
from tempfile import mkdtemp
import numpy as np

import nibabel as nib

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


def preprocess_mri(input_mri_path, apply_preproc_steps):

    temp_dir = mkdtemp()
    current_input = input_mri_path

    to_preprocess_dir = os.path.join(temp_dir, "to_preprocess")
    os.makedirs(to_preprocess_dir, exist_ok=True)

    image_destination = f"{temp_dir}/to_preprocess/image.nii.gz"
    shutil.copy(current_input, image_destination)

    if apply_preproc_steps:

        logger.info("Step 1: Preprocessing image with DISARM++.")
        preprocess_stgan_cmd = (
            f"/opt/toolkit/algos/DISARMpp/preprocessing/mri_prep.sh "
            f"{temp_dir}/to_preprocess "
            f"{temp_dir}/intermediaire "
            f"{temp_dir}/preprocessed "
        )

        run_cmd(preprocess_stgan_cmd)
        current_input = f"{temp_dir}/preprocessed"

    return current_input, temp_dir


def harmonize_image(input_mri, image_name, temp_dir, output_dir, save_preprocess):

    if save_preprocess:
        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True) 

        preprocess_output = os.path.join(preprocessed_dir, image_name)

        img_nii = nib.load(f"{input_mri}/registered_image.nii.gz")
        img = img_nii.get_fdata()
        img = 2 * (img - img.min()) / (img.max() - img.min()) - 1

        nib.save(nib.Nifti1Image(img, img_nii.affine, img_nii.header), preprocess_output)
        # shutil.copy(f"{temp_dir}/preprocessed/registered_image.nii.gz", preprocess_output)
        logger.info(f"Preprocessed image saved at: {preprocess_output}")

    logger.info("Step 2: Harmonizing image with DISAMR++ scanner free method.")
    harmonize_cmd = (
        f"python -u /opt/toolkit/algos/DISARMpp/code/inference.py "
        f"--input_dir {temp_dir}/preprocessed "
        f"--output_dir {temp_dir}/harmonized "
        f"--resume /opt/toolkit/algos/DISARMpp/checkpoint/trained_disarm++.pth "
        f"--gpu 0 "
        f"--mode scanner-free "
        f"--pre_trained_model "
        f"--temp_dir {temp_dir}"
    )
    run_cmd(harmonize_cmd)

    harmonized_image_path = f"{temp_dir}/harmonized/harm_registered_image.nii.gz"
    destination_path = os.path.join(output_dir, image_name)
    shutil.copy(harmonized_image_path, destination_path)
    logger.info(f"Harmonized image saved at: {destination_path}")


parser = argparse.ArgumentParser(description="Preprocess and harmonize MRI images using DISARM++.")
parser.add_argument("input_image", help="Path to the input MRI image.")
parser.add_argument("--apply_preproc_steps", type=bool, default=True,
                    help="Preprocessing steps to perform. Applied preproc steps: disarm default preproc. Default: True.")
parser.add_argument("--output_dir", default="output",
                    help="Directory to save the harmonized image.")
parser.add_argument("--save_preprocess", default=False,
                    help="Save preprocessed image.")

args = parser.parse_args()

image_name = os.path.basename(args.input_image)

os.makedirs(args.output_dir, exist_ok=True)

preprocessed_image, temp_dir = preprocess_mri(args.input_image, args.apply_preproc_steps)
harmonize_image(preprocessed_image, image_name, temp_dir, args.output_dir, args.save_preprocess)

# shutil.rmtree(temp_dir)
logger.info(f"Temporary directory {temp_dir} removed.")
