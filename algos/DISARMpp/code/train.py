from Utils import *
from Load_Dataset import *

from DISARM_class import *
import argparse

def main(opts):
    def train(opts):
    
        # data loader
        print('\n--- load dataset ---')
        dataset = data_multi_aug(opts) # Loader with augmentation (random elastic deformation)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
        
        print('\n--- set saver ---')
        saver = Saver(opts)
        
        # LOAD MODEL
        print('\n--- load model ---')
        model = DISARM(opts)
        model.setgpu(opts.gpu)
        if opts.resume is None:
            model.initialize()
            ep0 = -1
            total_it = 0
        else:
            ep0, total_it = model.resume(opts.resume)
        model.set_scheduler(opts, last_ep=ep0)
        ep0 += 1
        print('start the training at epoch %d'%(ep0))
        
        # TRAINING
        print('\n--- train ---')
        max_it = 200000
        
        stop_training = False
        
        for ep in range(ep0, opts.n_ep):
            for it, (images, c_org) in enumerate(train_loader):
                if images.size(0) != opts.batch_size:
                    continue
        
                # input data
                images = images.cuda(opts.gpu).detach()
                c_org = c_org.cuda(opts.gpu).detach()
        
                # update model
                if opts.isDcontent:
                    if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                        model.update_D_content(images, c_org)
                        continue
                    else:
                        model.update_D(images, c_org)
                        model.update_EG()
                else:
                    model.update_D(images, c_org)
                    model.update_EG()
              
                if (total_it+1) % opts.img_save_freq == 0:
                    saver.write_img(-1, model)
                if (total_it+1) % opts.model_save_freq == 0:
                    saver.write_model(-1,total_it, model)
                print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
                total_it += 1
                if total_it >= max_it:
                    saver.write_img(-1, model)
                    saver.write_model(-1,total_it, model)
                    stop_training = True  # Set the flag to stop training
                    break
              
            
        
            if stop_training:
                break  # Exit the outer loop
        
        # decay learning rate
        if opts.n_ep_decay > -1:
              model.update_lr()
        
        # save result image
        saver.write_img(ep, model)
        
        # Save network weights
        saver.write_model(ep, total_it, model)
    
    train(opts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')

    # data loader related
    parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    parser.add_argument('--num_domains', type=int, default=3)
    parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--input_dim', type=int, default=1, help='# of input channels')
    parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')

    # ouptput related
    parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
    parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
    parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')

    # training related
    parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
    parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_cls', type=float, default=3.0)
    parser.add_argument('--lambda_cls_G', type=float, default=10.0)
    parser.add_argument('--lambda_sf', type=float, default=7.0)
    parser.add_argument('--isDcontent', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    opts = parser.parse_args()

    main(opts)