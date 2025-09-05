import nibabel as nib
import numpy as np
import argparse

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--inpdir", help=""" Input path """)
parser.add_argument("--refdir", help=""" Header path """)
parser.add_argument("--outdir", help=""" Output path """)
parser.add_argument("--subject", help=""" Subject ID """)
args = parser.parse_args()

### Hedaer data read
head_intensity = nib.load(args.refdir)
head_data = head_intensity.get_fdata()
target_min = np.min(head_data)
target_max = np.max(head_data)

### Prediction read
img = nib.load(args.inpdir+ '/' + args.subject + '/' + args.subject+'_harm_whead.nii.gz')
pred_data = img.get_fdata()
src_min = np.min(pred_data)
src_max = np.max(pred_data)

### Rescale intensities
pred_rescaled= (target_max - target_min) * (pred_data - src_min) / (src_max - src_min) + target_min
pred_rescaled = np.clip(pred_rescaled, target_min, target_max)

### Save prediction image
b = nib.Nifti1Image(pred_rescaled.astype(np.float32), img.affine, img.header)
nib.save(b, args.outdir+ '/' + args.subject + '/' + args.subject+'_harm_whead_rescaled.nii.gz')


print("Done Rescaling intensities "+ args.subject)
