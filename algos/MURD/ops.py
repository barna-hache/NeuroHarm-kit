import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

factor, mode, uniform = pytorch_kaiming_weight_factor(uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
weight_regularizer = None
weight_regularizer_fully = None

##################################################################################
# Layers
##################################################################################

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, bias_init = tf.constant_initializer(0.0), sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=bias_init)
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias, bias_initializer=bias_init)

        return x

def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape =[x_shape[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

def gaussian_noise_layer(x, is_training=False):
    if is_training :
        noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32)
        return x + noise

    else :
        return x

##################################################################################
# Block
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):        
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)
            x = lrelu(x, 0.2)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)

        return x + x_init

def basic_block(x_init, channels, use_bias=True, sn=False, scope='basic_block') :
    with tf.variable_scope(scope) :
        x = lrelu(x_init, 0.2)
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        x = lrelu(x, 0.2)
        x = conv_avg(x, channels, use_bias=use_bias, sn=sn)

        shortcut = avg_conv(x_init, channels, use_bias=use_bias, sn=sn)

        return x + shortcut


def avg_conv(x, channels, use_bias=True, sn=False, scope='avg_conv') :
    with tf.variable_scope(scope) :
        x = avg_pooling(x, kernel=2, stride=2)
        x = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)

        return x

def conv_avg(x, channels, use_bias=True, sn=False, scope='conv_avg') :
    with tf.variable_scope(scope) :
        x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        x = avg_pooling(x, kernel=2, stride=2)

        return x

def expand_concat(x, z) :
    z = tf.reshape(z, shape=[z.shape[0], 1, 1, -1])
    z = tf.tile(z, multiples=[1, x.shape[1], x.shape[2], 1])  # expand
    x = tf.concat([x, z], axis=-1)

    return x

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.tanh(x)


##################################################################################
# Pooling & Resize
##################################################################################

def avg_pooling(x, kernel=2, stride=2, pad=0) :
    if pad > 0 :
        if (kernel - stride) % 2 == 0:
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else:
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

    return tf.layers.average_pooling2d(x, pool_size=kernel, strides=stride, padding='VALID')

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    return gap

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(real_logit, fake_logit):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit)
    return real_loss + fake_loss

def generator_loss(fake_logit):
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit)
    return fake_loss

def simple_gp(real_logit, fake_logit, real_images, fake_images, r1_gamma=10, r2_gamma=0):
    # Used in StyleGAN

    r1_penalty = 0
    r2_penalty = 0

    if r1_gamma != 0:
        real_loss = tf.reduce_mean(real_logit)  # FUNIT = reduce_mean, StyleGAN = reduce_sum
        real_grads = tf.gradients(real_loss, real_images)[0]

        r1_penalty = r1_gamma * tf.square(real_grads)
        # FUNIT didn't use 0.5

    if r2_gamma != 0:
        fake_loss = tf.reduce_mean(fake_logit)  # FUNIT = reduce_mean, StyleGAN = reduce_sum
        fake_grads = tf.gradients(fake_loss, fake_images)[0]

        r2_penalty = r2_gamma * tf.square(fake_grads)
        # FUNIT didn't use 0.5

    return r1_penalty + r2_penalty


def L1_loss(x, y):
    loss = tf.abs(x - y)

    return loss

def gradient_loss(x, y):
    dy_x, dx_x = tf.image.image_gradients(x)
    dy_y, dx_y = tf.image.image_gradients(y)
    loss = L1_loss(dy_x, dy_y) + L1_loss(dx_x, dx_y)
    return loss

def KL_divergence(mu) :
    # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, axis = -1)
    # loss = tf.reduce_mean(KL_divergence)
    mu_2 = tf.square(mu)
    loss = tf.reduce_mean(mu_2)

    return loss

def kl_loss(mu, logvar) :
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)
    
    return loss