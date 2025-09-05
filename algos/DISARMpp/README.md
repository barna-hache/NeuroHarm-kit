# DISARM++: Beyond Scanner-free harmonization

Variations in MRI data across different centers and scanners, due to unstandardized protocols, scanner- and acquisition-specific variabilities, can lead to significant inconsistencies in the biomarkers extracted from these datasets can affect the repeatability and the reproducibility of quantitative measure extracted from these datasets. Differences in hardware, software configurations, calibration procedures, maintenance practices, and operator experience can cause MRI scanners to produce images with varying contrast, brightness, and spatial resolution, i.e., voxel intensity distribution. This variability, particularly in multicenter studies, can introduce confounding effects that compromise the reliability of the results. 
We introduce DISARM++, a novel model for harmonizing 3D MR images by addressing inter-scanner variability. DISARM++ disentangles anatomical structure from scanner-specific information to generate scanner-free images. This approach preserves the original anatomical structure and biologically informative data, ensuring robust generalizability across various scanners.

With this repository, you can: 

1. **Use a pre-trained model** to harmonize MR images either to the *scanner-free* space or to a specific reference scanner (Gyroscan Intera).
2. **Train DISARM++ on your own dataset** to adapt harmonization to your specific imaging environment.

## Getting Started

Clone the repository:

```
git clone https://github.com/luca2245/DISARMpp_Harmonization.git
cd DISARMpp_Harmonization/code
```

## MRI Preprocessing
Preprocessing is required for both inference and training. The pipeline includes:

1. **Reorient to standard orientation** using [`fslreorient2std`](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/utilities/fslutils)
2. **Bias field correction** using [`FAST`](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/fast)
3. **Registration to MNI152-T1-1mm** using [`FLIRT`](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/)

To automate these preprocessing steps, utilize the provided `mri_prep.sh` script inside the `preprocessing` folder. 
Run the script using:

```
chmod +x mri_prep.sh
./mri_prep.sh /path/input_dir /path/output_dir
```
The script will process all images in the specified `input_dir` and save the preprocessed images to `output_dir`. The images must have extension `.nii.gz`.

### Options for DISARM++

- **Registration template:** Option `4` → `MNI152_T1_1mm.nii.gz`
- **FLIRT cost function:** Option `1` → Normalized Correlation Ratio  `normcorr`

If registration fails, try other cost functions provided by the script.

## Inference

### Pretrained model

```
python -u inference.py --input_dir /path/folder --output_dir /path/folder \ 
                       --resume /path/pretrained_model.pth --gpu 0 --mode scanner-free \
                       --pre_trained_model
```
**Arguments:**

- `input_dir` → Path to the folder containing the input .nii.gz MRI scans to be harmonized.
- `output_dir` → Path to the folder where harmonized images will be saved. Output files will retain their original filenames, prefixed with `harm_`
- `mode` → Harmonization mode. Choose between `scanner-free` or `reference`
- `pre_trained_model` → Flag to indicate usage of a pre-trained model. The pre-trained model is provided inside the `checkpoint` folder

### Self-trained model

```
python -u inference.py --input_dir /path/folder --output_dir /path/folder \ 
                       --resume /path/trained_model.pth --gpu 0 --mode reference \
                       --dataroot /path/data --domain_idx 2 
```
**Additional Arguments:**

- `dataroot` → Path to the dataset used during training. This directory should contain subfolders for each scanner domain.
- `domain_idx` → Required only in `reference` mode. Specifies the index of the scanner domain to use as the reference for harmonization. The script will select a reference image from the corresponding subfolder in `dataroot`. Not required in `scanner-free` mode.

## Training the model

### Dataset Setup

To set up the subfolders for training within `/path/dataset`, use the `create_scanner_folders()` function found in the `Load_Dataset.py` file. 
Detailed instructions on how to use this function are provided within the .py file itself.

### Training

```
python -u train.py --dataroot /path/data --batch_size 2 --num_domains 5 \ 
                   --input_dim 1 --result_dir model_savings --display_dir model_logs --d_iter 2 \ 
                   --n_ep 240000 --img_save_freq 500 --model_save_freq 1000 --isDcontent
```
**Key Parameters:**

- The image slices in the three dimensions (sagittal, axial, and coronal) from all reconstructions performed by the model during training are saved to `--result_dir` at intervals defined by `--img_save_freq`. 
- The model is saved to `--result_dir` at intervals specified by `--model_save_freq`.
- Use `--num_domains` to select the number of training scanner.

Our work benefit from [DISARM_Harmonization](https://github.com/luca2245/DISARM_Harmonization.git) and [DRIT](https://github.com/HsinYingLee/DRIT.git)
