from Utils import *
from Load_Dataset import *
from DISARM_class import *
import nibabel as nib

def main(opts):
    def inference(opts):
        os.makedirs(opts.output_dir, exist_ok=True)
                            
        opts.num_domains = 5
        opts.input_dim = 1
        opts.dis_scale = 3
        opts.dis_norm = 'None'
        opts.dis_spectral_norm = False
        opts.phase = 'train'
        
        # LOAD MODEL 
        print('\n--- load model ---')
        model = DISARM(opts)
        model.setgpu(opts.gpu)
        model.resume(opts.resume, train=False)
        model.eval()
        
        img_list, filenames_list = load_nifti_images_from_folder(opts.input_dir)
        
        if opts.mode == 'scanner-free':
            if opts.pre_trained_model == True:
                print('Harmonizing to scanner-free using the pretrained model...')
                z_rand = torch.tensor([[ 0.3440, -0.5120,  0.6164,  0.0888, -0.2443,  1.6746,  0.1562, -1.2279,
                 -0.3125,  0.4583,  1.5044, -1.2348,  1.2667,  2.1365, -1.3497, -0.8333]])
            else:
                print('Harmonizing to scanner-free...')
                z_rand = get_z_random(1, 16, random_type='gauss')
                
            har_data = transfer_img_list_to_scannerfree(img_list, z_rand, opts, model) 
        else:
            if opts.pre_trained_model == True:
                print('Harmonizing to reference using the pretrained model...')
                ref_img = nib.load('example_ref_image/ref_gyroscan.nii.gz').get_fdata()
                har_data = transfer_img_list_to_reference2(img_list, ref_img, 3, opts, model)
            else:
                print('Harmonizing to reference...')
                har_data = transfer_img_list_to_reference(img_list, opts.domain_idx, opts, model)
            
        save_preprocessed_images(opts.output_dir, [nib_tensor2img(har_data[i]) for i in range(len(har_data))], filenames_list, opts.temp_dir, prefix="harm_")
    
    inference(opts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script')

    # inference options
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing mr images to harmonize.')
    parser.add_argument('--dataroot', type=str, help='Path to the dataset used during training.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where harmonized images will be saved.')
    parser.add_argument('--resume', type=str, required=True, help='Directory containing the trained model.')
    parser.add_argument('--domain_idx', type=int, default=0, help='reference scanner index among the ones used for training')
    parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to use (default: 0).')
    parser.add_argument('--mode', type=str, default='scanner-free', 
                        choices=['scanner-free', 'reference'], 
                        help='Type of harmonization to perform. Options are "scanner-free" or "reference" (default: "scanner-free").')
    parser.add_argument('--pre_trained_model', action='store_true', help='Whether to use the pre-trained model.')
    parser.add_argument('--temp_dir', type=str, default='tmp_dir', help='pass temp_dir')

    opts = parser.parse_args()

    main(opts)