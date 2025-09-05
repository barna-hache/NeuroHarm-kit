"""
Script for preprocessing and harmonizing MRI images using IGUANE.

Run on CPU / GPU

"""

import os
import sys
import argparse
import logging
import shutil
import subprocess
import numpy as np
import nibabel as nib
from uuid import uuid4
from tempfile import gettempdir, mkdtemp
from tensorflow import convert_to_tensor
from tensorflow.config import list_physical_devices
from harmonization.model_architectures import Generator
from HD_BET.run import run_hd_bet


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define the temporary directory
TMP_DIR = gettempdir()

# Define the path to the MNI152 template
TEMPLATE_PATH = '/opt/toolkit/algos/IGUANE/preprocessing/MNI152_T1_1mm_brain.nii.gz'


if len(list_physical_devices('GPU'))>0:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")


def tmp_unique_path(extension='.nii.gz'):
    """Generate a unique temporary file path."""
    return os.path.join(TMP_DIR, f"{uuid4().hex}{extension}")


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
    

def indices_crop(data):
    """
    Determine cropping indices based on non-zero regions in the data.
    Returns:
        xmin, xsize, ymin, ysize, zmin, zsize
    """
    d1_1=0
    while d1_1<data.shape[0] and np.count_nonzero(data[d1_1,:,:])==0: d1_1+=1
    d1_2=0
    while d1_2<data.shape[0] and np.count_nonzero(data[-d1_2-1,:,:])==0: d1_2+=1
    d2_1=0
    while d2_1<data.shape[1] and np.count_nonzero(data[:,d2_1,:])==0: d2_1+=1
    d2_2=0
    while d2_2<data.shape[1] and np.count_nonzero(data[:,-d2_2-1,:])==0: d2_2+=1
    d3_1=0
    while d3_1<data.shape[1] and np.count_nonzero(data[:,:,d3_1])==0: d3_1+=1
    d3_2=0
    while d3_2<data.shape[1] and np.count_nonzero(data[:,:,-d3_2-1])==0: d3_2+=1
    
    # determine cropping
    if d1_1+d1_2 >= 22:
        if d1_1<11: xmin = d1_1
        elif d1_2<11: xmin = 182-160-d1_2
        else: xmin = 11
        xsize = 160
    else: xmin, xsize = 3,176
        
    if d2_1+d2_2 >= 26:
        if d2_1<13: ymin = d2_1
        elif d2_2<13: ymin = 218-192-d2_2
        else: ymin = 13
        ysize = 192
    else: ymin, ysize = 5,208
    
    if d3_1+d3_2 >= 22:
        if d3_1<11: zmin = d3_1
        elif d3_2<11: zmin = 182-160-d3_2
        else: zmin = 11
        zsize = 160
    else: zmin, zsize = 3,176
    return xmin, xsize, ymin, ysize, zmin, zsize



def preprocess_and_harmonize(in_mri, image_name, output_dir, available_gpu, save_preprocess, apply_preproc_steps):
    """
    Prétraitement et harmonisation d'une image IRM avec des étapes modulables.

    Paramètres :
    - in_mri : chemin vers l'image IRM d'entrée.
    - image_name : nom de l'image de sortie.
    - output_dir : répertoire de sortie.
    - available_gpu : booléen indiquant la disponibilité du GPU.
    - save_preprocess : booléen pour sauvegarder l'image prétraitée.
    - steps : liste des étapes à appliquer (parmi 'reorient', 'skullstrip', 'n4', 'mni', 'normalize', 'crop', 'harmonize').
    """

    temp_dir = mkdtemp()
    current_input = in_mri


    # Step 1: Reorientation
    if apply_preproc_steps:
        logger.info("Step 1: Reorientation")
        reorient_path = os.path.join(temp_dir, "reorient.nii.gz")
        run_cmd(f"/opt/tools/fsl509/bin/fslreorient2std {current_input} {reorient_path}")
        current_input = reorient_path

    # Step 2: Skull stripping
        logger.info("Step 2: Skull stripping")
        brain_native = os.path.join(temp_dir, "brain.nii.gz")
        if available_gpu:
            run_hd_bet(current_input, brain_native, bet=True)
        else:
            run_hd_bet(current_input, brain_native, bet=True, device='cpu', mode='fast', do_tta=False)
        mask_native = brain_native.replace('.nii.gz', '_mask.nii.gz')
        current_input = brain_native

    # Step 3: N4 bias field correction
        logger.info("Step 3: N4 bias field correction")
        n4native = os.path.join(temp_dir, "n4native.nii.gz")
        mask_option = f"-x {mask_native}" if mask_native else ""
        run_cmd(f"N4BiasFieldCorrection -i {current_input} {mask_option} -o {n4native}")
        current_input = n4native

    # Step 4: Registration to MNI space
        logger.info("Step 4: Registration to MNI space")
        n4mni = os.path.join(temp_dir, "n4mni.nii.gz")
        mni_mat = os.path.join(temp_dir, "mni.mat")
        run_cmd(f"/opt/tools/fsl509/bin/flirt -in {current_input} -ref {TEMPLATE_PATH} -omat {mni_mat} -interp trilinear -dof 6 -out {n4mni}")
        current_input = n4mni

        if mask_native:
            mask_mni = os.path.join(temp_dir, "mask_mni.nii.gz")
            run_cmd(f"/opt/tools/fsl509/bin/flirt -in {mask_native} -ref {TEMPLATE_PATH} -out {mask_mni} -init {mni_mat} -applyxfm -interp nearestneighbour")
            mask_native = mask_mni

    # Step 5: Median normalization
        logger.info("Step 5: Median normalization")
        median_mni = os.path.join(temp_dir, "median_mni.nii.gz")
        mri = nib.load(current_input)
        data = mri.get_fdata()
        if mask_native:
            mask = nib.load(mask_native).get_fdata() > 0
        else:
            mask = data > 0
        med = np.median(data[mask])
        data = data / med
        mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
        nib.save(mri, median_mni)
        current_input = median_mni

    # Step 6: Cropping
        logger.info("Step 6: Cropping")
        xmin, xsize, ymin, ysize, zmin, zsize = indices_crop(data)
        median_crop = os.path.join(temp_dir, "median_crop.nii.gz")
        run_cmd(f"/opt/tools/fsl509/bin/fslroi {current_input} {median_crop} {xmin} {xsize} {ymin} {ysize} {zmin} {zsize}")
        current_input = median_crop

    # Save preprocessed image
    if save_preprocess:
        preprocessed_dir = os.path.join(output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True) 

        preprocess_output = os.path.join(preprocessed_dir, image_name)

        shutil.copy(current_input, preprocess_output)
        logger.info(f"Preprocessed image saved at: {preprocess_output}")

    # Step 7: Harmonization
    logger.info("Step 7: Harmonization")
    generator = Generator()
    generator.load_weights('/opt/toolkit/algos/IGUANE/harmonization/iguane_weights.h5')
    mri = nib.load(current_input)
    data = mri.get_fdata() - 1
    mask = data > -1
    data[~mask] = 0
    data = np.expand_dims(data, axis=(0, 4))
    t = convert_to_tensor(data, dtype='float32')
    data = generator(t, training=False).numpy().squeeze()
    data = (data + 1) * 500
    data[~mask] = 0
    data = np.maximum(data, 0)
    mri = nib.Nifti1Image(data, affine=mri.affine, header=mri.header)
    output_mri = os.path.join(output_dir, image_name)
    nib.save(mri, output_mri)
    logger.info(f"Harmonized image saved at: {output_mri}")

    # Nettoyage du répertoire temporaire
    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} has been removed.")

parser = argparse.ArgumentParser(description="Preprocess and harmonize MRI images using IGUANE.")
parser.add_argument("input_image", help="Path to the input MRI image.")
parser.add_argument("--apply_preproc_steps", type=bool, default=True,
                    help="Preprocessing steps to perform. Applied preproc steps: default iguane steps. Default: True.")
parser.add_argument("--output_dir", default="output",
                    help="Directory to save the harmonized image.")
parser.add_argument("--save_preprocess", default=False,
                    help="Save preprocessed image.")

args = parser.parse_args()

image_name = os.path.basename(args.input_image)

os.makedirs(args.output_dir, exist_ok=True)

available_gpu = False
preprocess_and_harmonize(args.input_image, image_name, args.output_dir, available_gpu, args.save_preprocess, args.apply_preproc_steps)
