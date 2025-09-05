import os
import time
import numpy as np
from glob import glob
import tensorflow as tf
from ops import conv, instance_norm, lrelu, basic_block, resblock, global_avg_pooling, expand_concat, fully_connected, deconv, tanh, gaussian_noise_layer
from ops import generator_loss, L1_loss, gradient_loss, KL_divergence, discriminator_loss, simple_gp
from utils import check_folder, Image_data, save_images, return_images
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch # tf 1.13 or above

class MURD() :
    def __init__(self, sess, img_type):
        self.model_name = 'MURD'
        self.sess = sess
        self.img_type = img_type
        self.phase = 'train'
        self.dataset_name = 'ABCD'
        self.dataset_path = os.path.join('dataset', self.dataset_name, self.img_type, self.phase)
        self.label_list = [os.path.basename(x) for x in glob(self.dataset_path + '/*')]
        self.c_dim = len(self.label_list)
       
        self.img_height = 256
        self.img_width = 256
        self.img_ch = 3
        
        self.decay_iter = 50000

        self.iteration = 100000

        self.batch_size = 1
        self.print_freq = 100
        self.save_freq = 10000

        self.init_lr = 1e-4
        self.ema_decay = 0.999
        
        self.augment_flag = False
        self.decay_flag = True
        
        """ Weight """
        self.adv_weight = 1
        self.cont_weight = 10
        self.sty_weight = 10
        self.id_weight = 10
        self.cyc_weight = 10
        self.percep_weight = 1
        self.kl_weight = 1e-2
        self.r1_weight = 1
        self.sd_weight = 1

        """ Network parameters """
        self.ch = 32
        self.style_dim = 16
        self.n_layer = 4
        self.num_style = 1
        self.n_z = 64
        self.n_critic = 1
        
        """Output folders"""
        self.checkpoint_dir = os.path.join('checkpoint', self.img_type)
        self.sample_dir = os.path.join('samples', self.img_type)
        self.log_dir = os.path.join('log', self.img_type)
        
        check_folder(self.checkpoint_dir)
        check_folder(self.sample_dir)
        check_folder(self.log_dir)

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
    # Discriminator
    ##################################################################################
    
    def discriminator(self, x, scope="discriminator"):
        logit_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) :
            channel = self.ch
            x = conv(x, channels=channel * 2, kernel=4, stride=2, pad = 1, use_bias=True, scope='conv')
            x = lrelu(x, 0.2)
            
            channel = channel * 2

            x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='dis_conv_1')
            x = instance_norm(x, scope='instance_norm_1')
            x = lrelu(x, 0.2)
            
            channel = channel * 2
            
            x = conv(x, channel * 2, kernel=4, stride=1, pad=1, pad_type='reflect', scope='dis_conv_2')
            x = instance_norm(x, scope='instance_norm_2')
            x = lrelu(x, 0.2)

            x = global_avg_pooling(x)
            
            for i in range(self.c_dim):
                logit = fully_connected(x, units=1, use_bias=True, scope='logit_fc_' + str(i))
                logit_list.append(logit)

            return logit_list

    ##################################################################################
    # Model
    ##################################################################################
      
    def build_model(self):

        self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

        """ Input Image"""
        img_class = Image_data(self.img_height, self.img_width, self.img_ch, 
                               self.dataset_path, self.label_list, self.augment_flag)
        img_class.preprocess()

        dataset_num = len(img_class.image)
        print("Dataset number : ", dataset_num)

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        img_and_label = tf.data.Dataset.from_tensor_slices((img_class.image, img_class.label))

        gpu_device = '/gpu:0'
        img_and_label = img_and_label.apply(shuffle_and_repeat(dataset_num)).apply(
            map_and_batch(img_class.image_processing, self.batch_size, num_parallel_batches=16,
                          drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

        img_and_label_iterator = img_and_label.make_one_shot_iterator()

        self.x_real, label_org = img_and_label_iterator.get_next() # [bs, 256, 256, 3], [bs, 1]
        label_trg = tf.random_uniform(shape=tf.shape(label_org), minval=0, maxval=self.c_dim, dtype=tf.int32) # Target domain labels

        """ split """
        x_real = self.x_real

        with tf.device(gpu_device):
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                
                x_real_split = tf.split(x_real, num_or_size_splits=self.batch_size, axis=0)
                label_org_split = tf.split(label_org, num_or_size_splits=self.batch_size, axis=0)
                label_trg_split = tf.split(label_trg, num_or_size_splits=self.batch_size, axis=0)
                
                g_adv_loss = None
                g_cont_recon_loss = None
                g_sty_recon_loss = None
                g_id_loss = None
                g_cyc_loss = None
                g_percep_loss = None
                g_ca_loss = None
                g_sd_loss = None

                d_adv_loss = None
                d_simple_gp = None

                for each_bs in range(self.batch_size) :
                    """ Define Generator, Discriminator """
                    x_real_each = x_real_split[each_bs] # [1, 256, 256, 3]
                    label_org_each = tf.squeeze(label_org_split[each_bs], axis=[0, 1]) # [1, 1] -> []
                    label_trg_each = tf.squeeze(label_trg_split[each_bs], axis=[0, 1])
                    
                    random_style_code = tf.random_normal(shape=[self.batch_size, self.style_dim])
                    random_style_code2 = tf.random_normal(shape=[self.batch_size, self.style_dim])

                    random_style = tf.gather(self.style_generator(random_style_code), label_trg_each)
                    random_style2 = tf.gather(self.style_generator(random_style_code2), label_trg_each)
                    
                    x_content = self.content_encoder(x_real_each)
                    
                    x_fake = self.generator(x_content, random_style) # for adversarial 
                    x_fake2 = self.generator(x_content, random_style2)
                    x_fake2 = tf.stop_gradient(x_fake2)

                    x_real_each_style = tf.gather(self.style_encoder(x_real_each), label_org_each) # for cycle consistency
                    x_fake_style = tf.gather(self.style_encoder(x_fake), label_trg_each) # for style reconstruction
                    x_fake_content = self.content_encoder(x_fake)
                    
                    x_id = self.generator(x_content, x_real_each_style)
                    
                    x_cycle = self.generator(x_fake_content, x_real_each_style) # for cycle consistency
                    
                    real_logit = tf.gather(self.discriminator(x_real_each), label_org_each)
                    fake_logit = tf.gather(self.discriminator(x_fake), label_trg_each)
                    
                    """ Define loss """
                    
                    if each_bs == 0 :
                        g_adv_loss = self.adv_weight * generator_loss(fake_logit)
                        g_cont_recon_loss = self.cont_weight * L1_loss(x_content, x_fake_content)
                        g_sty_recon_loss = self.sty_weight * L1_loss(random_style, x_fake_style)
                        g_id_loss = self.id_weight * L1_loss(x_real_each, x_id)
                        g_cyc_loss = self.cyc_weight * L1_loss(x_real_each, x_cycle)
                        g_percep_loss = self.percep_weight * gradient_loss(x_real_each, x_cycle)
                        g_ca_loss = self.kl_weight * KL_divergence(x_content)
                        g_sd_loss = self.sd_weight * L1_loss(x_fake, x_fake2)

                        d_adv_loss = self.adv_weight * discriminator_loss(real_logit, fake_logit)
                        d_simple_gp = self.adv_weight * simple_gp(real_logit, fake_logit, x_real_each, x_fake, r1_gamma=self.r1_weight, r2_gamma=0.0)
                        
                g_adv_loss = tf.reduce_mean(g_adv_loss)
                g_cont_recon_loss = tf.reduce_mean(g_cont_recon_loss)
                g_sty_recon_loss = tf.reduce_mean(g_sty_recon_loss)
                g_id_loss = tf.reduce_mean(g_id_loss)
                g_cyc_loss = tf.reduce_mean(g_cyc_loss)
                g_percep_loss = tf.reduce_mean(g_percep_loss)

                d_adv_loss = tf.reduce_mean(d_adv_loss)
                d_simple_gp = tf.reduce_mean(tf.reduce_sum(d_simple_gp, axis=[1, 2, 3]))
                
                self.g_loss = g_adv_loss + g_sty_recon_loss + g_cont_recon_loss + g_id_loss + g_cyc_loss + g_percep_loss + g_ca_loss + g_sd_loss
                self.d_loss = d_adv_loss + d_simple_gp

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        E_vars = [var for var in t_vars if 'encoder' in var.name]
        EG_vars = [var for var in t_vars if 'style_generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        prev_g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=G_vars)
        prev_e_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=E_vars)
        prev_eg_optimizer = tf.train.AdamOptimizer(self.lr * 0.01, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=EG_vars)

        self.d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.d_loss, var_list=D_vars)

        with tf.control_dependencies([prev_g_optimizer, prev_e_optimizer, prev_eg_optimizer]):
            self.g_optimizer = self.ema.apply(G_vars)
            self.e_optimizer = self.ema.apply(E_vars)
            self.eg_optimizer = self.ema.apply(EG_vars)

        """" Summary """
        self.Generator_loss = tf.summary.scalar("g_loss", self.g_loss)
        self.Discriminator_loss = tf.summary.scalar("d_loss", self.d_loss)

        self.g_adv_loss = tf.summary.scalar("g_adv_loss", g_adv_loss)
        self.g_cont_recon_loss = tf.summary.scalar("g_cont_recon_loss", g_cont_recon_loss)
        self.g_sty_recon_loss = tf.summary.scalar("g_sty_recon_loss", g_sty_recon_loss)
        self.g_id_loss = tf.summary.scalar("g_id_loss", g_id_loss)
        self.g_cyc_loss = tf.summary.scalar("g_cyc_loss", g_cyc_loss)
        self.g_percep_loss = tf.summary.scalar("g_percep_loss", g_percep_loss)
        self.g_ca_loss = tf.summary.scalar("g_ca_loss", g_ca_loss)
        self.g_sd_loss = tf.summary.scalar("g_sd_loss", g_sd_loss)

        self.d_adv_loss = tf.summary.scalar("d_adv_loss", d_adv_loss)

        g_summary_list = [self.Generator_loss, self.g_adv_loss, self.g_sty_recon_loss, self.g_cont_recon_loss, self.g_id_loss, self.g_cyc_loss, self.g_percep_loss, self.g_kl_loss]
        d_summary_list = [self.Discriminator_loss, self.d_adv_loss]

        self.g_summary_loss = tf.summary.merge(g_summary_list)
        self.d_summary_loss = tf.summary.merge(d_summary_list)

        """ Result Image """
        def return_g_images(generator, image, code):
            x = generator(image, code)
            return x

        self.x_fake_list = []
        first_x_real = tf.expand_dims(self.x_real[0], axis=0)

        label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

        for _ in range(self.num_style):
            random_style_code = tf.truncated_normal(shape=[1, self.style_dim])
            self.x_fake_list.append(tf.map_fn(
                lambda c: return_g_images(self.generator,
                                          self.content_encoder(first_x_real, is_training=False),
                                          tf.gather(self.style_generator(random_style_code), c)),
                label_fix_list, dtype=tf.float32))

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=10)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_batch_id = checkpoint_counter
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr

        for idx in range(start_batch_id, self.iteration):
            if self.decay_flag :
                total_step = self.iteration
                current_step = idx
                decay_start_step = self.decay_iter

                if current_step < decay_start_step :
                    lr = self.init_lr
                else :
                    lr = self.init_lr * (total_step - current_step) / (total_step - decay_start_step)

                """ half decay """
                """
                if idx > 0 and (idx % decay_start_step) == 0 :
                    lr = self.init_lr * pow(0.5, idx // decay_start_step)
                """

            train_feed_dict = {
                self.lr : lr
            }

            # Update D
            _, d_loss, summary_str = self.sess.run([self.d_optimizer, self.d_loss, self.d_summary_loss], feed_dict = train_feed_dict)
            self.writer.add_summary(summary_str, counter)

            # Update G
            g_loss = None
            if (counter - 1) % self.n_critic == 0 :
                real_images, fake_images, _, _, _, g_loss, summary_str = self.sess.run([self.x_real, self.x_fake_list,
                                                                                  self.g_optimizer, self.e_optimizer, self.eg_optimizer,
                                                                                  self.g_loss, self.g_summary_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)
                past_g_loss = g_loss

            # display training status
            counter += 1
            if g_loss == None :
                g_loss = past_g_loss

            print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (idx, self.iteration, time.time() - start_time, d_loss, g_loss))

            if np.mod(idx+1, self.print_freq) == 0 :
                real_image = np.expand_dims(real_images[0], axis=0)
                save_images(real_image, [1, 1],
                            './{}/real_{:07d}.jpg'.format(self.sample_dir, idx+1))

                merge_fake_x = None

                for ns in range(self.num_style) :
                    fake_img = np.transpose(fake_images[ns], axes=[1, 0, 2, 3, 4])[0]

                    if ns == 0 :
                        merge_fake_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
                    else :
                        x = return_images(fake_img, [1, self.c_dim])
                        merge_fake_x = np.concatenate([merge_fake_x, x], axis=0)

                merge_fake_x = np.expand_dims(merge_fake_x, axis=0)
                save_images(merge_fake_x, [1, 1],
                            './{}/fake_{:07d}.jpg'.format(self.sample_dir, idx+1))

            if np.mod(counter - 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}adv_{}sty_{}cont_{}cyc_{}id_{}percep_{}kl".format(self.model_name, self.dataset_name,
                                                          self.adv_weight, self.sty_weight, self.cont_weight, 
                                                          self.cyc_weight, self.id_weight, self.percep_weight, 
                                                          self.kl_weight)    
    
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


if __name__ == '__main__':
    img_type = 'T2' # T1 or T2
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = MURD(sess, img_type)
        gan.build_model()       
        gan.train()
