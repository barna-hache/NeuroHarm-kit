# NeuroHarm-kit
An Open‑Source Framework for Benchmarking Deep‑Learning Harmonization of Multi‑Site T1‑Weighted MRI 

![Slide3](https://github.com/user-attachments/assets/cb214245-f83b-4286-b973-0abfacd82fac)

We developed an open‑source Python toolbox that unifies the end‑to‑end execution of 5 deep learning harmonization methods:
 - STGAN : Style transfer generative adversarial networks to harmonize multisite MRI to a single reference image to avoid overcorrection (https://doi.org/10.1002/hbm.26422)
 - HACA3: A unified approach for multi-site MR image harmonization (https://doi.org/10.1016/j.compmedimag.2023.102285)
 - MURD : Learning multi-site harmonization of magnetic resonance images without traveling human phantoms (https://doi.org/10.1038/s44172-023-00140-w)
 - IGUANe: A 3D generalizable CycleGAN for multicenter harmonization of brain MR images (https://doi.org/10.1016/j.media.2024.103388)
 - DISARM++: Beyond scanner-free harmonization (https://doi.org/10.48550/arXiv.2505.03715)

## 1. Prerequisites

To use this toolbox, check the following requirements :
 - Install docker via the official website : https://docs.docker.com/get-started/
 - In order for the docker to detect the GPU on your device, install the NVIDIA Container Toolkit via the official website : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html. You can still use some of the algorithms without any GPU or NVIDIA Container Toolkit installed 

Note : All the preprocessing pipelines are already implemented according to each algorithms. All the pretrained weights are already present in the project. There is no need to train/finetune these models.

## 2. Installation

To intall the toolbox on your device, check the following steps:
 - Download the project at following link : google drive link
 - Unzip the project in the folder of your choice
 - In the bash terminal, navigate to the project folder: `cd path/to/the/project/folder`
 - With docker, build the docker image : `docker build -t neuroharmo_kit .`

Note : The unzipped project is 9 GiB (6.8 when zipped) but the built docker image is ~40GiB because each algorithm has its own environment. 

## 3. Usage

To use the following commands, please navigate to the project folder : `cd path/to/the/project/folder`

### STGAN example command:
```
./neuroharmo_toolkit.sh stgan \
path/to/nifti_image_to_harmonize.nii.gz \
path/to/output_folder
```
Options:\
`--apply_preproc_steps` [True/False]: If you already applied the preprocessed stpes to the image, skip the preprocessing pipeline applied by the toolkit by turning this option to False. Default is True.\
`--save_preprocess` [True/False]: If you want to save the preprocessed image before harmonization to a folder “preprocessed” in your outpu folder created by the toolbox. Default is True.

### HACA3 example command:
```
./neuroharmo_toolkit.sh haca3 \
path/to/nifti_image_to_harmonize.nii.gz \
path/to/output_folder
```
Options:\
`--apply_preproc_steps` [True/False]\
`--save_preprocess` [True/False]\
`--theta [float] –-theta [float]` : Choose the target contrast. Default is 10.0 20.0

### MURD example command:
```
./neuroharmo_toolkit.sh murd \
path/to/nifti_image_to_harmonize.nii.gz \
path/to/output_folder
```
Options:\
`--apply_preproc_steps` [True/False]\
`--n_axial_slices` [int] : As MURD harmonize slice by slice you can choose the number of central axial slice you want. Default is 200 to harmonize all the brain.

Note : There is no --apply_preproc_steps (always True) option available because MURD works with png images.

### IGUANe example command:
```
./neuroharmo_toolkit.sh iguane \
path/to/nifti_image_to_harmonize.nii.gz \
path/to/output_folder
```
Options:\
`--apply_preproc_steps` [True/False]\
`--save_preprocess` [True/False]

### DISARM++ example command:
```
./neuroharmo_toolkit.sh disarmpp \
path/to/nifti_image_to_harmonize.nii.gz \
path/to/output_folder
```
Options:\
`--apply_preproc_steps` [True/False]\
`--save_preprocess` [True/False]
