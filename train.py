import argparse
import numpy as np
import tensorflow as tf
import time
import os

import mnist_input
import cifar10_input

from model import ConcreteVae
from utils import plot_2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=50000,
                        help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--sample_step', type=int, default=5000,
                        help='Plot a sample this often')
    parser.add_argument('--checkpoint_step', type=int, default=10000,
                        help='Save a model checkpoint this often')
    parser.add_argument('--anneal_rate', type=float, default=3e-5,
                        help='The anneal rate of the gumbel temperature')
    parser.add_argument('--min_temp', type=float, default=0.5,
                        help='The minimum gumbel temperature')
    parser.add_argument('--initial_temp', type=float, default=1.,
                        help='The initial gumbel temperature')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='The initial learning rate')
    parser.add_argument('--cont_dim', type=int, default=1,
                        help='The number of continuous latent dimensions')
    parser.add_argument('--discrete_dim', type=int, default=10,
                        help='The number of categories in the discrete'
                        'latent dimension')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='The data where the training data is located')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='The dataset (mnist or cifar10)')
    args = parser.parse_args()

    current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    checkpoint_dir = 'checkpoint/' + current_time
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    sample_dir = 'sample/' + current_time
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    return train(num_iters=args.iters,
                 batch_size=args.batch_size,
                 checkpoint_step=args.checkpoint_step,
                 initial_temp=args.initial_temp,
                 min_temp=args.min_temp,
                 anneal_rate=args.anneal_rate,
                 learning_rate=args.learning_rate,
                 checkpoint_dir=checkpoint_dir,
                 sample_step=args.sample_step,
                 sample_dir=sample_dir,
                 cont_dim=args.cont_dim,
                 discrete_dim=args.discrete_dim,
                 data_dir=args.data_dir,
                 dataset=args.dataset)


def train(num_iters=20000, batch_size=100, checkpoint_step=10000,
          initial_temp=1., min_temp=0.5, anneal_rate=0.00003,
          learning_rate=1e-3, checkpoint_dir='checkpoint',
          sample_step=5000, sample_dir='sample', cont_dim=2,
          discrete_dim=10, data_dir=None, dataset='mnist'):
    if data_dir is None:
        raise Exception('The directory which contains the training data'
                        'must be passed in using --data_dir')

    if dataset == 'mnist':
        input_, _ = mnist_input.inputs(True, data_dir, batch_size)
        input_ = tf.reshape(input_, [-1, 28, 28, 1])
        data_shape = input_.get_shape().as_list()[1:]
    elif dataset == 'cifar10':
        input_, _ = cifar10_input.inputs(True, data_dir, batch_size)
        input_ = input_ + .5
        data_shape = input_.get_shape().as_list()[1:]
    else:
        raise Exception('The dataset must be "mnist" or "cifar10"')

    vae = ConcreteVae(input_=input_, cont_dim=cont_dim,
                      discrete_dim=discrete_dim)

    sess = tf.Session()
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    temp = initial_temp
    for i in range(num_iters):
        _, loss, cat_kl, cont_kl, recon = \
            sess.run([vae.optimizer, vae.loss, vae.discrete_kl, vae.normal_kl,
                     vae.reconstruction], {vae.tau: temp,
                                           vae.learning_rate: learning_rate})

        if i % checkpoint_step == 0:
            path = saver.save(sess, checkpoint_dir + '/model.ckpt')
            print('Model saved at iteration {} in checkpoint {}'
                  .format(i, path))
        # TODO: support plotting samples with more than a single continuous
        # and discrete dimension
        if i % sample_step == 0 and cont_dim + discrete_dim == 11:
            plot_2d(sess, sample_dir=sample_dir, step=i,
                    num_categories=discrete_dim, vae=vae, shape=data_shape)
            print('Sample generated at step {}'.format(i))
        if i % 1000 == 0:
            temp = np.maximum(initial_temp * np.exp(-anneal_rate * i),
                              min_temp)
            learning_rate *= 0.9
            print('Temperature updated to {}\n'.format(temp) +
                  'Learning rate updated to {}'.format(learning_rate))
        if i % 1000 == 0:
            print('Iteration {}\nLoss: {}\n'.format(i, loss) +
                  'Categorical KL: {}\n'.format(np.mean(cat_kl)) +
                  'Continuous KL: {}\n'.format(np.mean(cont_kl)) +
                  'Recon: {}'.format(np.mean(recon)))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
