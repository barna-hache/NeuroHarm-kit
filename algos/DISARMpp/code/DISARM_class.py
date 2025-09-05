import torch
import torch.nn as nn
import numpy as np
from Modules import *
from Utils import *
import os
from torch.optim import lr_scheduler
from torch.autograd import Variable


class DISARM(nn.Module):
  def __init__(self, opts):
    super(DISARM, self).__init__()
    self.opts = opts
    lr = 0.0001
    lr_dcontent = lr/2.5
    self.nz = 16

    self.dis1 = Scanner_Discriminator(opts.input_dim, norm=opts.dis_norm, sn=opts.dis_spectral_norm,  
                                      c_dim=opts.num_domains)
    self.dis2 = Scanner_Discriminator(opts.input_dim, norm=opts.dis_norm, sn=opts.dis_spectral_norm, 
                                      c_dim=opts.num_domains)
    self.enc_c = Anatomy_Encoder(opts.input_dim)
    
    self.enc_a = Scanner_Encoder(opts.input_dim, output_nc=self.nz, c_dim=opts.num_domains, 
          norm_layer=None, nl_layer=get_non_linearity(layer_type='lrelu'))
    self.gen = Generator(opts.input_dim, c_dim=opts.num_domains, nz=self.nz)

    self.dis1_opt = torch.optim.Adam(self.dis1.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.dis2_opt = torch.optim.Adam(self.dis2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    
    self.disContent = Anatomy_Discriminator(c_dim=opts.num_domains)
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)

    self.cls_loss = nn.BCEWithLogitsLoss()

  def initialize(self):
    self.dis1.apply(gaussian_weights_init)
    self.dis2.apply(gaussian_weights_init)
    self.disContent.apply(gaussian_weights_init)
    self.gen.apply(gaussian_weights_init)
    self.enc_c.apply(gaussian_weights_init)
    self.enc_a.apply(gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.dis1_sch = get_scheduler(self.dis1_opt, opts, last_ep)
    self.dis2_sch = get_scheduler(self.dis2_opt, opts, last_ep)
    self.disContent_sch = get_scheduler(self.disContent_opt, opts, last_ep)
    self.enc_c_sch = get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = get_scheduler(self.gen_opt, opts, last_ep)

  def update_lr(self):
    self.dis1_sch.step()
    self.dis2_sch.step()
    self.disContent_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def setgpu(self, gpu):
    self.gpu = gpu
    self.dis1.cuda(self.gpu)
    self.dis2.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    self.disContent.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def test_reference_transfer(self, image, image_trg, c_trg):
    self.z_content = self.enc_c.forward(image)
    self.mu, self.logvar = self.enc_a.forward(image_trg, c_trg) 
    std = self.logvar.mul(0.5).exp_()
    eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
    self.z_attr = eps.mul(std).add_(self.mu)
    output = self.gen.forward(self.z_content, self.z_attr, c_trg)
    return output

  def test_scannerfree_transfer(self, image, z_random):
    self.z_content = self.enc_c.forward(image)
    c_trg = np.zeros((image.size(0),self.opts.num_domains))
    c_trg = torch.FloatTensor(c_trg).cuda()
    output = self.gen.forward(self.z_content, z_random, c_trg)
    return output

  def forward(self):
    if not self.input.size(0)%2 == 0:
      print("Need to be even QAQ")
      input()
    half_size = self.input.size(0)//2
    self.real_A = self.input[0:half_size]
    self.real_B = self.input[half_size:]
    c_org_A = self.c_org[0:half_size]
    c_org_B = self.c_org[half_size:]

    # Get encoded anatomy
    self.real_img = torch.cat((self.real_A, self.real_B),0)
    self.z_content = self.enc_c.forward(self.real_img)
    self.z_content_a, self.z_content_b = torch.split(self.z_content, half_size, dim=0)

    # Get encoded scanner effect (mu and sigma)
    self.mu, self.logvar = self.enc_a.forward(self.real_img, self.c_org)
    std = self.logvar.mul(0.5).exp_()
    eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
    self.z_attr = eps.mul(std).add_(self.mu)
        
    self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, half_size, dim=0)
    self.z_random = self.get_z_random(half_size, self.nz, 'gauss')

    # Cross translation #1
    # row1: (images_a = real_A, images_b1 = fake_B_encoded, images_b2 = fake_B_random, images_a4 = fake_AA_encoded, images_a3 = fake_A_recon)
    # row2: (images_b = real_B, images_a1 = fake_A_encoded, images_a2 = fake_A_random, images_b4 = fake_BB_encoded, images_b3 = fake_B_recon)
    input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b),0)
    input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a),0)
    input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random),0)
    input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
    input_c_forA = torch.cat((c_org_A, c_org_A, c_org_A), 0)
    input_c_forB = torch.cat((c_org_B, c_org_B, c_org_B), 0)
    output_fakeA = self.gen.forward(input_content_forA, input_attr_forA, input_c_forA)
    output_fakeB = self.gen.forward(input_content_forB, input_attr_forB, input_c_forB)
    self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
    self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # Get reconstructed encoded anatomy
    self.fake_encoded_img = torch.cat((self.fake_A_encoded, self.fake_B_encoded),0)
    self.z_content_recon = self.enc_c.forward(self.fake_encoded_img)
    self.z_content_recon_b, self.z_content_recon_a = torch.split(self.z_content_recon, half_size, dim=0)

    # Get reconstructed encoded scanner effect
    self.mu_recon, self.logvar_recon = self.enc_a.forward(self.fake_encoded_img, self.c_org)
    std_recon = self.logvar_recon.mul(0.5).exp_()
    eps_recon = self.get_z_random(std_recon.size(0), std_recon.size(1), 'gauss')
    self.z_attr_recon = eps_recon.mul(std_recon).add_(self.mu_recon)
        
    self.z_attr_recon_a, self.z_attr_recon_b = torch.split(self.z_attr_recon, half_size, dim=0)

    # Cross translation #2
    self.fake_A_recon = self.gen.forward(self.z_content_recon_a, self.z_attr_recon_a, c_org_A)
    self.fake_B_recon = self.gen.forward(self.z_content_recon_b, self.z_attr_recon_b, c_org_B)

    # Image to display during training
    self.image_display = torch.cat((self.real_A[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),                                     self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(),                                     self.real_B[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(),                                     self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

    self.fake_random_img = torch.cat((self.fake_A_random, self.fake_B_random), 0)
    self.mu2, _= self.enc_a.forward(self.fake_random_img, self.c_org)
    self.mu2_a, self.mu2_b = torch.split(self.mu2, half_size, 0)
    
    # For Scanner-Free space Loss
    self.free_attr = torch.zeros(1,self.opts.num_domains).cuda(self.gpu)
    self.fake_A_free = self.gen.forward(self.z_content_a, self.z_random, self.free_attr)
    self.fake_B_free = self.gen.forward(self.z_content_b, self.z_random, self.free_attr)

    self.mu_A_free, self.logvar_A_free = self.enc_a.forward(self.fake_A_free, self.free_attr)
    std_A_free = self.logvar_A_free.mul(0.5).exp_()
    eps_A_free = self.get_z_random(std_A_free.size(0), std_A_free.size(1), 'gauss')
    self.z_attr_A_free = eps_A_free.mul(std_A_free).add_(self.mu_A_free)

    self.mu_B_free, self.logvar_B_free = self.enc_a.forward(self.fake_B_free, self.free_attr)
    std_B_free = self.logvar_B_free.mul(0.5).exp_()
    eps_B_free = self.get_z_random(std_B_free.size(0), std_B_free.size(1), 'gauss')
    self.z_attr_B_free = eps_B_free.mul(std_B_free).add_(self.mu_B_free)


  def update_D_content(self, image, c_org):
    self.input = image
    self.z_content = self.enc_c.forward(self.input)
    self.disContent_opt.zero_grad()
    pred_cls = self.disContent.forward(self.z_content.detach())
    loss_D_content = self.cls_loss(pred_cls, c_org)
    loss_D_content.backward()
    self.disContent_loss = loss_D_content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 1.0)
    self.disContent_opt.step()

  def update_D(self, image, c_org):
    self.input = image
    self.c_org = c_org
    self.forward()

    self.dis1_opt.zero_grad()
    self.D1_gan_loss, self.D1_cls_loss = self.backward_D(self.dis1, self.input, self.fake_encoded_img)
    nn.utils.clip_grad_norm_(self.dis1.parameters(), 1.0)
    self.dis1_opt.step()

    self.dis2_opt.zero_grad()
    self.D2_gan_loss, self.D2_cls_loss = self.backward_D(self.dis2, self.input, self.fake_random_img)
    nn.utils.clip_grad_norm_(self.dis2.parameters(), 1.0)
    self.dis2_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake, pred_fake_cls = netD.forward(fake.detach())
    pred_real, pred_real_cls = netD.forward(real)
    loss_D_gan = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
        out_fake = nn.functional.sigmoid(out_a)
        out_real = nn.functional.sigmoid(out_b)
        all0 = torch.zeros_like(out_fake).cuda(self.gpu)
        all1 = torch.ones_like(out_real).cuda(self.gpu)
        ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
        ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
        loss_D_gan += ad_true_loss + ad_fake_loss

    loss_D_cls = self.cls_loss(pred_real_cls, self.c_org)
    loss_D = loss_D_gan + self.opts.lambda_cls * loss_D_cls
    loss_D.backward()
    return loss_D_gan, loss_D_cls

  def update_EG(self):
    #update G, Ec, Ea
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    nn.utils.clip_grad_norm_(self.enc_c.parameters(), 1.0)
    nn.utils.clip_grad_norm_(self.enc_a.parameters(), 1.0)
    nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()
    #update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.forward() # call forward() to using new network parameters to compute variables
    self.backward_G_alone()
    nn.utils.clip_grad_norm_(self.enc_c.parameters(), 1.0)
    nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):
    # Adversarial loss for generator
    loss_G_GAN_content = self.backward_G_GAN_content(self.z_content)

    # Ladv for generator
    pred_fake, pred_fake_cls = self.dis1.forward(self.fake_encoded_img)
    loss_G_GAN = 0
    for out_a in pred_fake:
        outputs_fake = nn.functional.sigmoid(out_a)
        all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
        loss_G_GAN += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

    # Classification loss
    loss_G_cls = self.cls_loss(pred_fake_cls, self.c_org) * self.opts.lambda_cls_G

    # Self-Reconstruction and Cross-cycle reconstruction losses
    loss_G_L1_self = torch.mean(torch.abs(self.input - torch.cat((self.fake_AA_encoded, self.fake_BB_encoded), 0))) * self.opts.lambda_rec
    loss_G_L1_cc = torch.mean(torch.abs(self.input - torch.cat((self.fake_A_recon, self.fake_B_recon), 0))) * self.opts.lambda_rec

    # KL loss
    loss_kl_zc = self._l2_regularize(self.z_content) * 0.01
    kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
    loss_kl_za = torch.sum(kl_element).mul_(-0.5) * 0.01
    
    # Scanner-Free Loss
    loss_sf = torch.mean( torch.abs( self.z_attr_A_free - self.z_attr_B_free ) ) * self.opts.lambda_sf

    loss_G = loss_G_GAN + loss_G_cls + loss_G_L1_self + loss_G_L1_cc + loss_kl_zc + loss_kl_za + loss_sf
    loss_G += loss_G_GAN_content
    loss_G.backward(retain_graph=True)

    self.gan_loss = loss_G_GAN.item()
    self.gan_cls_loss = loss_G_cls.item()
    self.gan_loss_content = loss_G_GAN_content.item()
    self.kl_loss_zc = loss_kl_zc.item()
    self.kl_loss_za = loss_kl_za.item()
    self.l1_self_rec_loss = loss_G_L1_self.item()
    self.l1_cc_rec_loss = loss_G_L1_cc.item()
    self.G_loss = loss_G.item()
    self.loss_sf = loss_sf.item()

  def backward_G_GAN_content(self, data):
    pred_cls = self.disContent.forward(data)
    loss_G_content = self.cls_loss(pred_cls, 1-self.c_org)
    return loss_G_content

  def backward_G_alone(self):
    # Adversarial loss for generator
    pred_fake, pred_fake_cls = self.dis2.forward(self.fake_random_img)
    loss_G_GAN2 = 0
    for out_a in pred_fake:
        outputs_fake = nn.functional.sigmoid(out_a)
        all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
        loss_G_GAN2 += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

    # Classification loss
    loss_G_cls2 = self.cls_loss(pred_fake_cls, self.c_org) * self.opts.lambda_cls_G

    # Latent regression loss
    loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 8
    loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 8

    loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2 + loss_G_cls2
    loss_z_L1.backward()
    self.l1_recon_z_loss = loss_z_L1_a.item() + loss_z_L1_b.item()
    self.gan2_loss = loss_G_GAN2.item()
    self.gan2_cls_loss = loss_G_cls2.item()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def update_lr(self):
    self.dis1_sch.step()
    self.dis2_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A).detach()
    images_b = self.normalize_image(self.real_B).detach()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach()
    images_a2 = self.normalize_image(self.fake_A_random).detach()
    images_a3 = self.normalize_image(self.fake_A_recon).detach()
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b2 = self.normalize_image(self.fake_B_random).detach()
    images_b3 = self.normalize_image(self.fake_B_recon).detach()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()

    z_slice_index = 85
    x_slice_index = 15
    img_a = np.rot90(tensor2img(images_a[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_b =  np.rot90(tensor2img(images_b[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_a1 =  np.rot90(tensor2img(images_a1[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_a2 =  np.rot90(tensor2img(images_a2[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_a3 =  np.rot90(tensor2img(images_a3[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_a4 =  np.rot90(tensor2img(images_a4[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_b1 =  np.rot90(tensor2img(images_b1[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_b2 =  np.rot90(tensor2img(images_b2[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_b3 =  np.rot90(tensor2img(images_b3[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    img_b4 =  np.rot90(tensor2img(images_b4[0:1, ::])[:, :, x_slice_index, :, :], axes=(2, 3))
    row1 = np.concatenate((img_a, img_b1, img_b2, img_a4, img_a3),3)
    row2 = np.concatenate((img_b, img_a1, img_a2, img_b4, img_b3),3)
    final_montage1 = np.concatenate((row1, row2), axis = 2)
    
    z_slice_index = 85
    img_a = np.rot90(tensor2img(images_a[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_b =  np.rot90(tensor2img(images_b[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_a1 =  np.rot90(tensor2img(images_a1[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_a2 =  np.rot90(tensor2img(images_a2[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_a3 =  np.rot90(tensor2img(images_a3[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_a4 =  np.rot90(tensor2img(images_a4[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_b1 =  np.rot90(tensor2img(images_b1[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_b2 =  np.rot90(tensor2img(images_b2[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_b3 =  np.rot90(tensor2img(images_b3[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    img_b4 =  np.rot90(tensor2img(images_b4[0:1, ::])[:, :, :, z_slice_index, :], axes=(2, 3))
    row1 = np.concatenate((img_a, img_b1, img_b2, img_a4, img_a3),3)
    row2 = np.concatenate((img_b, img_a1, img_a2, img_b4, img_b3),3)
    final_montage2 = np.concatenate((row1, row2), axis = 2)
    
    z_slice_index = 85
    img_a = np.rot90(tensor2img(images_a[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_b =  np.rot90(tensor2img(images_b[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_a1 =  np.rot90(tensor2img(images_a1[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_a2 =  np.rot90(tensor2img(images_a2[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_a3 =  np.rot90(tensor2img(images_a3[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_a4 =  np.rot90(tensor2img(images_a4[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_b1 =  np.rot90(tensor2img(images_b1[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_b2 =  np.rot90(tensor2img(images_b2[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_b3 =  np.rot90(tensor2img(images_b3[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    img_b4 =  np.rot90(tensor2img(images_b4[0:1, ::])[:, :, :, :, z_slice_index], axes=(2, 3))
    row1 = np.concatenate((img_a, img_b1, img_b2, img_a4, img_a3),3)
    row2 = np.concatenate((img_b, img_a1, img_a2, img_b4, img_b3),3)
    final_montage3 = np.concatenate((row1, row2), axis = 2)
    
    return final_montage1, final_montage2, final_montage3 

  def normalize_image(self, x):
    return x[:,0:1,:,:, :]


  def save(self, filename, ep, total_it):
    state = {
             'dis1': self.dis1.state_dict(),
             'dis2': self.dis2.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'dis1_opt': self.dis1_opt.state_dict(),
             'dis2_opt': self.dis2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # Weight
    if train:
        self.dis1.load_state_dict(checkpoint['dis1'])
        self.dis2.load_state_dict(checkpoint['dis2'])
        self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # Optimizer
    if train:
        self.dis1_opt.load_state_dict(checkpoint['dis1_opt'])
        self.dis2_opt.load_state_dict(checkpoint['dis2_opt'])
        self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
        self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
        self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
        self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

