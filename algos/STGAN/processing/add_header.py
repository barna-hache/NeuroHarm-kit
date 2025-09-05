import nibabel as nib
import numpy as np
import argparse

#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--inpdir", help=""" Input path """)
parser.add_argument("--subject", help=""" Subject ID """)
args = parser.parse_args()


head_affine = nib.load(args.inpdir + '/' + args.subject + '_src.nii.gz')
img = nib.load(args.inpdir+'/'+args.subject+'_harm.nii.gz')
data = img.get_fdata()

b = nib.Nifti1Image(data.astype(np.float32), head_affine.affine, head_affine.header)
nib.save(b, args.inpdir+'/'+args.subject+'_harm_whead.nii.gz')


print("Done Reattaching Header to "+ args.subject)
