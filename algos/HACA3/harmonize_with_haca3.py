"""
Script for preprocessing and harmonizing MRI images using HACA3.

Run on GPU only
"""

import argparse
import os
import shutil
import subprocess
import numpy as np
import logging
from tempfile import mkdtemp
import nibabel as nib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define the path to the MNI152 template
TEMPLATE_PATH = "/opt/toolkit/algos/HACA3/MNI_192x224x192_1mm_brain.nii.gz"

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
    """
    Preprocess the MRI image based on specified steps.

    Parameters:
    - input_mri_path: Path to the input MRI image.
    - steps: List of preprocessing steps to perform.

    Returns:
    - Tuple containing:
        - Path to the preprocessed MRI image.
        - Path to the temporary directory used during preprocessing.
    """
    temp_dir = mkdtemp()
    current_input = input_mri_path

    if apply_preproc_steps:
        logger.info("Step 1: N4BiasFieldCorrection")
        n4_output = os.path.join(temp_dir, "n4native.nii.gz")
        n4_cmd = f"N4BiasFieldCorrection -i {current_input} -o {n4_output}"
        run_cmd(n4_cmd)
        current_input = n4_output

        logger.info("Step 2: Reorient to standard")
        reorient_output = os.path.join(temp_dir, "reorient.nii.gz")
        reorient_cmd = f"/opt/tools/fsl509/bin/fslreorient2std {current_input} {reorient_output}"
        run_cmd(reorient_cmd)
        current_input = reorient_output

        logger.info("Step 3: Brain extraction with ROBEX")
        robex_brain = os.path.join(temp_dir, "robex_brain.nii.gz")
        robex_mask = os.path.join(temp_dir, "robex_brain_mask.nii.gz")
        robex_cmd = f"sh /opt/tools/ROBEX/runROBEX.sh {current_input} {robex_brain} {robex_mask}"
        run_cmd(robex_cmd)
        current_input = robex_brain

        logger.info("Step 4: Register to MNI space")
        mni_output = os.path.join(temp_dir, "MNI_mri.nii.gz")
        mni_matrix = os.path.join(temp_dir, "MNI_matrix.mat")

        # Step 4a: Compute transformation matrix from the brain-extracted image
        flirt_cmd_matrix = f"flirt -in {current_input} -ref {TEMPLATE_PATH} -omat {mni_matrix} -interp trilinear -dof 6"
        run_cmd(flirt_cmd_matrix)

        # Step 4b: Apply transformation to the reoriented image
        flirt_cmd_apply = f"flirt -in {reorient_output} -ref {TEMPLATE_PATH} -out {mni_output} -init {mni_matrix} -applyxfm -interp trilinear"
        run_cmd(flirt_cmd_apply)

        current_input = mni_output

    return current_input, temp_dir

def harmonize_image(input_image, image_name, temp_dir, output_dir, theta, save_preprocess):
    """
    Harmonize the MRI image using HACA3.

    Parameters:
    - input_image: Path to the input MRI image.
    - image_name: Name of the image file.
    - temp_dir: Temporary directory used during processing.
    - output_dir: Directory to save the harmonized image.
    - theta: List of theta values for harmonization.
    - save_preprocess: Boolean indicating whether to save the preprocessed image.
    """
    if save_preprocess:
        logger.info("Step 4: Intensity normalization (applied only if you save preprocessed image)")
        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True) 

        preprocess_output = os.path.join(preprocessed_dir, image_name)

        img_nii = nib.load(input_image)
        img = img_nii.get_fdata()

        p99 = np.percentile(img.flatten(), 95)
        img = img / (p99 + 1e-5)
        img = np.clip(img, a_min=0.0, a_max=5.0)

        nib.save(nib.Nifti1Image(img, img_nii.affine, img_nii.header), preprocess_output)

        # shutil.copy(os.path.join(temp_dir, "MNI_mri.nii.gz"), preprocess_output)
        logger.info(f"Preprocessed image saved at: {preprocess_output}")

    theta_str = ' '.join(map(str, theta))
    cmd = (
        f"singularity exec --nv -e -B /data/in /opt/toolkit/algos/HACA3/haca3_v1.0.9.sif haca3-test "
        f"--in-path {input_image} "
        f"--target-theta {theta_str} "
        f"--harmonization-model /opt/toolkit/algos/HACA3/pretrained_weights/harmonization_public.pt "
        f"--fusion-model /opt/toolkit/algos/HACA3/pretrained_weights/fusion.pt "
        f"--out-path {os.path.join(temp_dir, image_name)} "
        f"--intermediate-out-dir {temp_dir}"
    )
    run_cmd(cmd)

    harmonized_file = os.path.join(temp_dir, f"{image_name.split('.')[0]}_harmonized_fusion.nii.gz")
    destination_file = os.path.join(output_dir, image_name)
    shutil.copy(harmonized_file, destination_file)

    logger.info(f"Harmonized image saved at: {destination_file}")


parser = argparse.ArgumentParser(description="Preprocess and harmonize MRI images using HACA3.")
parser.add_argument("input_image", help="Path to the input MRI image.")
parser.add_argument("--apply_preproc_steps", type=bool, default=True,
                    help="Preprocessing steps to perform. Applied preproc steps: 'n4', 'mni'. Default: True.")
parser.add_argument("--theta", nargs="+", type=float, default=[10.0, 20.0],
                    help="Theta values for harmonization. Default: [10.0, 20.0] (T1w).")
parser.add_argument("--output_dir", default="output",
                    help="Directory to save the harmonized image. Default: 'output'.")
parser.add_argument("--save_preprocess", type=bool, default=False,
                    help="Whether to save the preprocessed image. Default: False.")

args = parser.parse_args()

image_name = os.path.basename(args.input_image)
os.makedirs(args.output_dir, exist_ok=True)

preprocessed_image, temp_dir = preprocess_mri(args.input_image, args.apply_preproc_steps)
harmonize_image(preprocessed_image, image_name, temp_dir, args.output_dir, args.theta, args.save_preprocess)

shutil.rmtree(temp_dir)
logger.info(f"Temporary directory {temp_dir} has been removed.")
