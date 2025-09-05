import os
import numpy as np
from glob import glob
import tensorflow as tf
from ops import conv, instance_norm, lrelu, basic_block, resblock, global_avg_pooling, expand_concat, fully_connected, deconv, tanh, gaussian_noise_layer
from utils import check_folder, save_test_images, load_test_image, save_images
from tqdm import tqdm
from natsort import natsorted

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class MURD() :
    def __init__(self, sess, img_type, test_mode, L_path_img, L_dir_out, refer_type = None):
        self.model_name = 'MURD'
        self.sess = sess
        self.img_type = img_type
        self.test_mode = test_mode
        self.dataset_path = os.path.join('/opt/toolkit/algos/MURD/dataset', self.img_type, self.test_mode)
        self.label_list = [os.path.basename(x) for x in glob(self.dataset_path + '/*')]
        self.c_dim = len(self.label_list)
        
        self.img_height = 256
        self.img_width = 256
        self.img_ch = 3
        
        if refer_type != None:
            self.refer_type = refer_type
            print(glob(os.path.join('/opt/toolkit/algos/MURD/dataset', self.img_type, 'reference', self.refer_type, '*.png')))
            self.refer_img_dir = glob(os.path.join('/opt/toolkit/algos/MURD/dataset', self.img_type, 'reference', self.refer_type, '*.png'))[0]
        self.checkpoint_dir = os.path.join('/opt/toolkit/algos/MURD/weights', self.img_type)
        
        self.ema_decay = 0.999
        
        """ Weight """
        self.adv_weight = 1
        self.cont_weight = 10
        self.sty_weight = 10
        self.id_weight = 10
        self.cyc_weight = 10
        self.percep_weight = 1
        self.kl_weight = 1e-2
        self.r1_weight = 1
        
        self.ds_weight = 1
        
        """ Network parameters """
        self.ch = 32
        self.style_dim = 16
        self.n_layer = 4
        self.n_z = 64
        self.n_critic = 1
        
        """Output folder"""
        self.result_dir = os.path.join('/opt/toolkit/algos/MURD/results', self.img_type)

        #test
        self.L_path_img = L_path_img
        self.L_dir_out = L_dir_out

    ##################################################################################
    # Generator
    ##################################################################################
   
    def content_encoder(self, x_init, is_training=True, scope="content_encoder"):
        channel = self.ch * 2
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = conv(x_init, channels=channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_1x1')
            x = instance_norm(x, scope='ins_norm')
            x = lrelu(x, 0.2)

            for i in range(self.n_layer//2) :
                x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = lrelu(x, 0.2)
                channel = channel * 2

            for i in range(self.n_layer):
                x = resblock(x, channels=channel, use_bias=True, scope='inter_pre_resblock_' + str(i))
                
            x = gaussian_noise_layer(x, is_training)
            
            return x
    
    def style_encoder(self, x_init, scope="style_encoder"):
        channel = self.ch
        style_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):           
            x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv')

            for i in range(self.n_layer):
                x = basic_block(x, channel * 2, scope='basic_block_' + str(i))
                channel = channel * 2
            
            x = lrelu(x, 0.2)
            x = global_avg_pooling(x)

            for i in range(self.c_dim):
                style = fully_connected(x, units=self.n_z, use_bias=True, scope='style_fc_' + str(i))
                style_list.append(style)

            return style_list

    def style_generator(self, latent_z, scope='style_generator'):
        channel = self.ch * pow(2, self.n_layer//2)
        style_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = latent_z

            for i in range(self.n_layer + self.n_layer // 2):
                x = fully_connected(x, units=channel, use_bias=True, scope='fc_' + str(i))
                x = lrelu(x, 0.2)

            for i in range(self.c_dim) :
                style = fully_connected(x, units=self.n_z, use_bias=True, scope='style_fc_' + str(i))
                style_list.append(style)

            return style_list # [c_dim,], style_list[i] = [bs, 64]
        
    def generator(self, content, style, scope="generator"):
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) :
            channel = self.ch*2*np.power(2, self.n_layer//2) + self.n_z
            x = expand_concat(content, style)
            
            for i in range(self.n_layer) :
                x = resblock(x, channel, scope='resblock_' + str(i))

            for i in range(self.n_layer//2):
                x = deconv(x, channel // 2, kernel=4, stride=2, scope='deconv_' + str(i))
                x = instance_norm(x, scope='instance_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)
            
            return x

    ##################################################################################
    # Model
    ##################################################################################   
    
    def build_model(self):

        self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

        if self.test_mode == 'reference_guided':
            """ Test """

            def return_g_images(generator, image, code):
                x = generator(image, code)
                return x

            self.custom_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='custom_image')
            self.refer_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='refer_image')

            label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

            self.refer_fake_image = tf.map_fn(
                lambda c : return_g_images(self.generator,
                                            self.content_encoder(self.custom_image, is_training=False),
                                            tf.gather(self.style_encoder(self.refer_image), c)),
                label_fix_list, dtype=tf.float32)

        elif self.test_mode == 'site_guided':
            """ Test """

            def return_g_images(generator, image, code):
                x = generator(image, code)
                return x

            self.custom_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='custom_image')
            label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

            random_style_code = tf.truncated_normal(shape=[1, self.style_dim], stddev=2.0)
            self.custom_fake_image = tf.map_fn(
                lambda c : return_g_images(self.generator,
                                            self.content_encoder(self.custom_image, is_training=False),
                                            tf.gather(self.style_generator(random_style_code), c)),
                label_fix_list, dtype=tf.float32)

    @property
    def model_dir(self):
        return "{}".format(self.model_name)    
    
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
    def test(self):
        if self.test_mode == 'site_guided':
            self.site_guided_test()
        elif self.test_mode == 'reference_guided':
            self.reference_guided_test()


    def reference_guided_test(self):
        tf.global_variables_initializer().run()
        test_MRI_files = glob('/opt/toolkit/algos/MURD/dataset/{}/{}/{}/*.png'.format(self.img_type, self.test_mode, self.label_list[2]))
        # print(test_Siemens_files) --> ['./dataset/T1/reference_guided/Siemens/test_Siemens_010.png', ...]

        test_files = [self.L_path_img]
        #refer_image = load_test_image(self.refer_img_dir, self.img_width, self.img_height, self.img_ch)
        refer_image = load_test_image(self.refer_img_dir, self.img_width, self.img_height, self.img_ch)

        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name or 'encoder' in var.name or 'style_gen' in var.name]

        shadow_G_vars_dict = {}
        for g_var in G_vars:
            shadow_G_vars_dict[self.ema.average_name(g_var)] = g_var

        self.saver = tf.train.Saver(shadow_G_vars_dict)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.test_mode)
        check_folder(self.result_dir)

        im_path = self.L_dir_out
        check_folder(im_path)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for test_type in test_files:
            fake_MRI_imgs = []
            for sample_file in tqdm(natsorted(test_type)):
                print("Processing image: " + sample_file)
                sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
                fake_img = self.sess.run(self.refer_fake_image, feed_dict={self.custom_image: sample_image, self.refer_image: refer_image})
                fake_img = np.transpose(fake_img, axes=[1, 0, 2, 3, 4])[0]
                fake_MRI_imgs.append(np.expand_dims(fake_img[2], axis=0))
            
            fake_MRI_imgs = np.concatenate(fake_MRI_imgs, axis=0)
            MRI_idx = np.zeros((self.img_height, self.img_width, fake_MRI_imgs.shape[0] + 2))
            MRI_img = np.zeros((self.img_height, self.img_width, fake_MRI_imgs.shape[0] + 2))

            for k in range(fake_MRI_imgs.shape[0]):
                MRI_idx[:, :, k:k+3] = MRI_idx[:, :, k:k+3] + np.ones((self.img_height, self.img_width, self.img_ch))
                MRI_img[:, :, k:k+3] = MRI_img[:, :, k:k+3] + fake_MRI_imgs[k]
            MRI_imgs = np.true_divide(MRI_img, MRI_idx)

            for m in range(MRI_imgs.shape[-1]):
                idx = os.path.basename(test_type[0]).split('_')
                if m < 9:
                    sym = '00'
                else:
                    sym = '0'
                filename = os.path.basename(test_type[0]).replace(idx[-1], sym + str(m + 1) + '.png')
                img_path = os.path.join(im_path, '{}'.format(filename))
                save_test_images(MRI_imgs[:, :, m], img_path)
