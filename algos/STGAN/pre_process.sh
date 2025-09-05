#!/bin/sh

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Assign the input and output directories to variables
input_directory=$1
output_directory=$2
register_ref=/opt/toolkit/algos/STGAN/processing/register_ref/mni_icbm152_t1_tal_nlin_sym_09a_brain.nii.gz

# Create Directory for storing processed files
mkdir -p $output_directory/matrices

# Process all files in directory
# for input_file in "$input_directory"/*; do
for input_file in "$input_directory"/*normalized*.nii.gz; do

    if [ -f "$input_file" ]; then

        file_name=$(basename "$input_file" .nii.gz)

        # Register the Image
        echo "Registering $file_name ..."
        flirt -in $input_directory/${file_name}.nii.gz -ref ${register_ref} -out ${output_directory}/${file_name} \
        -dof 9 -cost mutualinfo -omat ${output_directory}/matrices/${file_name}.xfm
        
        # Zero-pad Images
        echo "Resizing $file_name ..."
        python /opt/toolkit/algos/STGAN/processing/PadNifti.py --inpdir ${output_directory}/ --outdir ${output_directory}/ --subject ${file_name}.nii.gz
    fi
done

echo "Processed All files"
echo "Files Saved at $output_directory"
