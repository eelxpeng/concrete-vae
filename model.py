import numpy as np
import tensorflow as tf

from utils import sample_normal, sample_gumbel, kl_categorical, \
    kl_normal

Bernoulli = tf.contrib.distributions.Bernoulli
slim = tf.contrib.slim

"""
Concrete Variational Autoencoder

The following code was used as a guide:
https://github.com/EmilienDupont/vae-concrete
https://github.com/ericjang/gumbel-softmax
"""


class ConcreteVae():
    def __init__(self, input_, cont_dim=2, discrete_dim=0,
                 filters=[32, 64], hidden_dim=1024, model_name="ConcreteVae"):
        """
        Constructs a Variational Autoencoder that supports continuous and
        discrete dimensions. Currently only one discrete dimension is
        supported.

        Args:
        input_        the input tensor
        cont_dim      the number of continuous latent dimensions
        discrete_dim  the number of categories in the discrete latent dimension
        filters       the number of filters for each convolution
        hidden_dim    the dimension of the fully-connected hidden layer between
                          the convolutions and the latent variable
        model_name    the name of the model
        """
        self.input_ = input_
        input_shape = input_.get_shape().as_list()
        print('Input shape {}'.format(input_shape))

        self.model_name = model_name

        # Build the encoder
        # According to karpathy, generative models work better when
        # they discard pooling layers in favor of larger strides
        # (https://cs231n.github.io/convolutional-networks/#pool)
        net = slim.conv2d(self.input_, filters[0], kernel_size=5, stride=2,
                          padding='SAME')
        net = slim.conv2d(net, filters[1], kernel_size=5, stride=2,
                          padding='SAME')
        # Use dropout to reduce overfitting
        # net = slim.dropout(net, 0.9)
        net = slim.flatten(net)

        # Sample from the latent distribution
        q_z_mean = slim.fully_connected(net, cont_dim, activation_fn=None)
        q_z_log_var = slim.fully_connected(net, cont_dim, activation_fn=None)
        # TODO: support multiple categorical variables
        q_category_logits = slim.fully_connected(net, discrete_dim,
                                                 activation_fn=None)
        q_category = tf.nn.softmax(q_category_logits)
        self.q_z_mean = q_z_mean
        self.q_z_log_var = q_z_log_var
        self.q_category = q_category

        self.continuous_z = sample_normal(q_z_mean, q_z_log_var)
        self.tau = tf.Variable(5.0, name="temperature")
        self.category = sample_gumbel(q_category_logits, self.tau)
        self.z = tf.concat([self.continuous_z, self.category], axis=1)

        # Build the decoder
        net = tf.reshape(self.z, [-1, 1, 1, cont_dim + discrete_dim])
        net = slim.conv2d_transpose(net, filters[1], kernel_size=5,
                                    stride=2, padding='SAME')
        net = slim.conv2d_transpose(net, filters[0], kernel_size=5,
                                    stride=2, padding='SAME')
        net = slim.conv2d_transpose(net, input_shape[3], kernel_size=5,
                                    padding='VALID')
        net = slim.flatten(net)
        # TODO: figure out the whole logits and Bernoulli dist vs MSE thing
        # Do not include the batch size in creating the final layer
        self.logits = slim.fully_connected(net, np.product(input_shape[1:]),
                                           activation_fn=None)
        print('Output shape {}'.format(self.logits.get_shape()))
        p_x = Bernoulli(logits=self.logits)
        self.p_x = p_x

        self.loss = self._vae_loss()
        self.learning_rate = tf.Variable(1e-3, name="learning_rate")
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate) \
            .minimize(self.loss, var_list=slim.get_model_variables())

    def _vae_loss(self):
        # TODO: should the KL divergences be weighted to factor in the
        # number of variables. For isntance kl_normal uses reduce_sum,
        # should it be divded by the number of normal variables
        discrete_kl = kl_categorical(self.q_category)
        normal_kl = kl_normal(self.q_z_mean, self.q_z_log_var)
        # reconstruction = tf.reduce_sum(
        #    self.p_x.log_prob(slim.flatten(self.input_)), 1)

        d = (slim.flatten(self.input_) - self.logits)
        d2 = tf.multiply(d, d) * 2.0
        reconstruction = -tf.reduce_sum(d2, 1)

        self.discrete_kl = discrete_kl
        self.normal_kl = normal_kl
        self.reconstruction = reconstruction

        elbo = reconstruction - discrete_kl - normal_kl
        return tf.reduce_mean(-elbo)
