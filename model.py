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
    def __init__(self, cont_dim=2, discrete_dim=0, input_shape=(28, 28, 1),
                 filters=[32, 64], hidden_dim=1024, model_name="ConcreteVae"):
        """
        Constructs a Variational Autoencoder that supports continuous and
        discrete dimensions. Currently only one discrete dimension is
        supported.

        Args:
        cont_dim      the number of continuous latent dimensions
        discrete_dim  the number of categories in the discrete latent dimension
        input_shape   the shape of the input (rows, cols, channels)
        filters       the number of filters for each convolution
        hidden_dim    the dimension of the fully-connected hidden layer between
                          the convolutions and the latent variable
        model_name    the name of the model
        """

        self.input_shape = input_shape
        self.model_name = model_name

        self.x = tf.placeholder(tf.float32,
                                [None, np.product(self.input_shape)])

        # Build the encoder
        net = tf.reshape(self.x, [-1, 28, 28, 1])
        net = slim.conv2d(net, filters[0], kernel_size=5, stride=1,
                          padding='SAME')
        net = slim.max_pool2d(net, kernel_size=2, stride=2, padding='SAME')
        net = slim.conv2d(net, filters[1], kernel_size=5, stride=1,
                          padding='SAME')
        net = slim.max_pool2d(net, kernel_size=2, stride=2, padding='SAME')
        net = slim.flatten(net, scope='flatten1')
        net = slim.fully_connected(net, hidden_dim)
        # TODO: consider dropout to reduce overfitting
        # net = slim.dropout(net, 0.9)

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

        continuous_z = sample_normal(q_z_mean, q_z_log_var)
        self.tau = tf.Variable(5.0, name="temperature")
        category = sample_gumbel(q_category_logits, self.tau)
        z = tf.concat([continuous_z, category], axis=1)
        self.z = z

        # Build the decoder
        net = slim.fully_connected(self.z, hidden_dim)
        net = slim.fully_connected(net, 7 * 7 * filters[1])
        net = tf.reshape(net, [-1, 7, 7, filters[1]])
        net = slim.conv2d_transpose(net, filters[1], kernel_size=5 * 2,
                                    stride=1, padding='SAME')
        net = slim.conv2d_transpose(net, filters[0], kernel_size=5 * 2,
                                    stride=1, padding='SAME')
        net = slim.conv2d_transpose(net, input_shape[2], kernel_size=5 * 2,
                                    padding='VALID')
        net = slim.flatten(net)
        logits = slim.fully_connected(net, np.product(self.input_shape),
                                      activation_fn=None)
        p_x = Bernoulli(logits=logits)
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
        reconstruction = tf.reduce_sum(self.p_x.log_prob(self.x), 1)

        self.discrete_kl = discrete_kl
        self.normal_kl = normal_kl
        self.reconstruction = reconstruction

        elbo = reconstruction - discrete_kl - normal_kl
        return tf.reduce_mean(-elbo)
