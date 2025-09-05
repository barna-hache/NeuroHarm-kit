#!/usr/bin/env bash
export FSLOUTPUTTYPE=NIFTI_GZ
export FSLMULTIFILEQUIT=FALSE
echo $FSLOUTPUTTYPE
echo $FSLMULTIFILEQUIT
. $FSLDIR/etc/fslconf/fsl.sh

# CHECK FOR THE CORRECT NUMBER OF INPUTS
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_folder> <intermediate_folder> <final_output_folder>"
    exit 1
fi

# GET INPUT FOLDER, INTERMEDIATE FOLDER, AND FINAL OUTPUT FOLDER
input_folder="$1"
intermediate_folder="$2"
final_output_folder="$3"

# PROMPT USER TO SELECT THE REFERENCE IMAGE
# echo "Select the reference image for registration:"
# echo "1) MNI152_T1_0.5mm.nii.gz"
# echo "2) MNI152_T1_0.7mm.nii.gz"
# echo "3) MNI152_T1_0.8mm.nii.gz"
# echo "4) MNI152_T1_1mm.nii.gz"
# echo "5) MNI152_T1_2mm.nii.gz"

# read -p "Enter the number corresponding to your choice (1-5): " ref_choice
ref_choice=4

# SET REFERENCE IMAGE BASED ON USER CHOICE
case $ref_choice in
    1) ref_image="$FSLDIR/data/standard/MNI152_T1_0.5mm.nii.gz" ;;
    2) ref_image="$FSLDIR/data/standard/MNI152_T1_0.7mm.nii.gz" ;;
    3) ref_image="$FSLDIR/data/standard/MNI152_T1_0.8mm.nii.gz" ;;
    4) ref_image="$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz" ;;
    5) ref_image="$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz" ;;
    *) echo "Invalid choice. Exiting."; exit 1 ;;
esac

# PROMPT USER TO SELECT THE COST FUNCTION FOR FLIRT
# echo "Select the cost function for FLIRT registration:"
# echo "1) Normalised Correlation Ratio (intra-modal)"
# echo "2) Least Squares (intra-modal)"
# echo "3) Correlation Ratio (inter-modal)"
# echo "4) Mutual Information (inter-modal)"
# echo "5) Normalised Mutual Information (inter-modal)"

# read -p "Enter the number corresponding to your choice (1-5): " cost_choice
cost_choice=1

# SET COST FUNCTION BASED ON USER CHOICE
case $cost_choice in
    1) cost_function="normcorr" ;;
    2) cost_function="leastsq" ;;
    3) cost_function="corratio" ;;
    4) cost_function="mutualinfo" ;;
    5) cost_function="normmi" ;;
    *) echo "Invalid choice. Using default: Normalised Correlation Ratio -> normcorr"; cost_function="normcorr" ;;
esac

# CREATE INTERMEDIATE AND FINAL OUTPUT FOLDERS IF THEY DON'T EXIST
[ ! -d "$intermediate_folder" ] && mkdir -p "$intermediate_folder"
[ ! -d "$final_output_folder" ] && mkdir -p "$final_output_folder"

# OPTIONS FOR FAST COMMAND
common_options="-t 1 -n 3 -H 0.1 -I 4 -l 20.0 --nopve -B -b"

# LOOP THROUGH ALL .nii.gz FILES IN THE INPUT FOLDER
for image in "$input_folder"/*.nii.gz; do
    # Check if the pattern matched any files
    if [ ! -e "$image" ]; then
        echo "No .nii.gz files found in the input folder."
        exit 1
    fi

    # DEFINE OUTPUT PREFIXES FOR INTERMEDIATE RESULTS
    image_basename=$(basename "${image%.*}")
    output_prefix="$intermediate_folder/output_$image_basename"
    restored_output="$intermediate_folder/output_${image_basename%.*}_restore"
    reoriented_image="$intermediate_folder/reoriented_$image_basename"

    # Print the image being processed
    echo "Processing image: $image_basename"

    # ORIENT THE IMAGE TO THE STANDARD ORIENTATION AND SAVE IN INTERMEDIATE FOLDER
    fslreorient2std "$image" "$reoriented_image"

    # APPLY BIAS-FIELD CORRECTION WITH FAST
    $FSLDIR/bin/fast $common_options -o "$output_prefix" "$reoriented_image"

    # REGISTER IMAGE TO THE SELECTED REFERENCE IMAGE WITH FLIRT
    $FSLDIR/bin/flirt -in "${restored_output}.nii.gz" -ref "$ref_image" -out "$final_output_folder/registered_$image_basename" -cost $cost_function -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear
    #$FSLDIR/bin/flirt -in "${restored_output}.nii.gz" -ref "$ref_image" -out "$final_output_folder/registered_$image_basename" -cost $cost_function -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12 -interp trilinear
    #$FSLDIR/bin/flirt -in "${restored_output}.nii.gz" -ref "$ref_image" -out "$final_output_folder/registered_$image_basename" -cost $cost_function -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6 -interp trilinear
    #$FSLDIR/bin/flirt -in "${restored_output}.nii.gz" -ref "$ref_image" -out "$final_output_folder/registered_$image_basename" -cost $cost_function -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 6 -interp trilinear
    
    echo "Image $image_basename processed and saved as registered_$image_basename"

done
