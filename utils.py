import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

EPSILON = 1e-20


def sample_normal(z_mean, z_log_var):
    """
    Sample from the normal distribution N~(z_mean, sqrt(exp(z_log_var)))

    Args:
    z_mean     the mean of the normal distribution
    z_log_var  the log variance of the normal distribution
    """
    shape = tf.shape(z_log_var)
    epsilon = tf.random_normal(shape, mean=0., stddev=1.)
    std_dev = tf.sqrt(tf.exp(z_log_var))
    return z_mean + epsilon * std_dev


def sample_gumbel(category_logits, temperature=0.5):
    """
    Sample from the Gumbel-Softmax distribution

    Args:
    category_logits  (batch_size, categories) unnormalized log-probs
    temperature      non-negative scalar
    """
    shape = tf.shape(category_logits)
    uniform = tf.random_uniform(shape, minval=0, maxval=1)
    gumbel = -tf.log(-tf.log(uniform + EPSILON) + EPSILON)
    logit = (category_logits + gumbel) / temperature
    return tf.nn.softmax(logit)


def kl_normal(z_mean, z_log_var):
    kl = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) -
                              tf.exp(z_log_var), 1)
    return kl


def kl_categorical(dist):
    num_categories = dist.get_shape().as_list()[1]
    print('Distribution contains {} categories'.format(num_categories))
    kl = dist * (tf.log(dist + EPSILON) - tf.log(1. / num_categories))
    kl = tf.reduce_sum(kl, 1)
    return kl


def plot_2d(sess, sample_dir, step, num_categories, vae, shape=(28, 28, 1)):
    """
    TODO: this plots one image at a time, batch for speed
    """
    nx = 10
    ny = 10
    x_values = np.linspace(-2, 2, nx)
    y_values = range(num_categories)

    height = shape[0]
    width = shape[1]
    channels = shape[2]

    if channels == 1:
        canvas = np.empty((height * ny, width * nx))
    else:
        canvas = np.empty((height * ny, width * nx, channels))
    for i, yi in enumerate(y_values):
        for j, xi in enumerate(x_values):
            category = np.zeros([1, num_categories])
            category[0][yi] = 1
            continuous_z = [[xi]]
            new_z = np.concatenate([continuous_z, category], axis=1)
            np_x = sess.run([vae.p_x.mean()], {vae.z: new_z})
            if channels == 1:
                canvas[i * height:(i + 1) * height,
                       j * width:(j + 1) * width] = \
                    np_x[0].reshape(height, width)
            else:
                canvas[i * height:(i + 1) * height,
                       j * width:(j + 1) * width] = \
                    np_x[0].reshape(shape)
    plt.figure(figsize=(20, 20))
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()
    plt.savefig(sample_dir + '/' + str(step))
