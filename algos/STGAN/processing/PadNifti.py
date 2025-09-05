import nibabel as nib
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--inpdir")
parser.add_argument("--outdir")
parser.add_argument("--subject")
args=parser.parse_args()
pad_dir=args.outdir
src_dir=args.inpdir
it=args.subject

path_orig=src_dir+it
img=nib.load(path_orig)
img1_=img.get_fdata()

## Set minimum intensity to 0
img1 = img1_ - img1_.min()

data=np.zeros((256,256,256), dtype=np.int16)
x,y,z=img1.shape
data[128-x//2:129+x//2,128-y//2:129+y//2,128-z//2:129+z//2]=img1
res=nib.Nifti1Image(data,affine=img.affine,header=img.header)
nib.save(res,pad_dir+it)

