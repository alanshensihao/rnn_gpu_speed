from gpu_speed_plot import line_plot

import os
import tensorflow as tf

import numpy as np
import random

import time
import pickle

from tensorflow.contrib.rnn import GRUCell, static_rnn
from tensorflow.contrib.cudnn_rnn import CudnnGRU


def set_random_seeds():
    os.environ['PYTHONHASHSEED'] = '0'
    tf.set_random_seed(1234)
    np.random.seed(42)
    random.seed(7)


def create_rnn_graph(x, var_scope, n_units, cudnn=False, reuse=False):
    with tf.variable_scope(var_scope, reuse=reuse):
        xav_init = tf.contrib.layers.xavier_initializer()
        zeros_init = tf.zeros_initializer()

        xT = tf.transpose(x, perm=[1, 0, 2])

        if cudnn:
            cudnn_gru_layer = CudnnGRU(num_layers=1, num_units=n_units, bias_initializer=zeros_init,
                                       kernel_initializer=xav_init, dtype=tf.float32)

            rnn_outputs, states = cudnn_gru_layer(xT)

        else:
            x_split = tf.unstack(xT)

            RNN_CELL = GRUCell(num_units=n_units, reuse=reuse, bias_initializer=zeros_init,
                               kernel_initializer=xav_init, dtype=tf.float32)

            rnn_outputs, states = static_rnn(RNN_CELL, x_split, dtype=tf.float32)

        logits = tf.layers.dense(rnn_outputs[-1], units=1, activation=None, kernel_initializer=xav_init,
                                 kernel_regularizer=None)

    return logits


def main():
    sequence_dim = 500
    sequence_len = 50
    n_samples = 256

    n_epochs = 50

    n_rnn_units = 1024

    run_configs = [{'use_cuda': True, 'use_cudnn_rnn': False, 'var_scope': 'tf_cuda_base_rnn', 'label': 'CUDA'},
                   {'use_cuda': True, 'use_cudnn_rnn': True, 'var_scope': 'tf_cuda_cudnn_rnn', 'label': 'CUDA+Cudnn'},
                   {'use_cuda': False, 'use_cudnn_rnn': False, 'var_scope': 'tf_no_cuda', 'label': 'No GPU'}]

    epoch_times = np.zeros((len(run_configs), n_epochs))
    epoch_losses = np.zeros((len(run_configs), n_epochs))

    x_data = np.random.normal(size=(n_samples, sequence_len, sequence_dim)).astype(np.float32)
    y_data = np.random.normal(size=(n_samples, 1)).astype(np.float32)

    plot_speed_info = list()
    plot_loss_info = list()

    for run_count, cnfg in enumerate(run_configs):
        tf.reset_default_graph()

        set_random_seeds()

        x = tf.placeholder(tf.float32, [None, sequence_len, sequence_dim])
        y = tf.placeholder(tf.float32, [None, 1])

        var_scope = cnfg['var_scope']

        rnn_out = create_rnn_graph(x, var_scope=var_scope, n_units=n_rnn_units, cudnn=cnfg['use_cudnn_rnn'])
        rnn_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=rnn_out)

        opt = tf.train.AdamOptimizer()
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope)
        opt_step = opt.minimize(rnn_loss, var_list=trainable_vars)

        init = tf.global_variables_initializer()

        gpu_count = 1 if cnfg['use_cuda'] else 0

        config = tf.ConfigProto(
            device_count={'GPU': gpu_count}
        )

        with tf.Session(config=config) as sess:
            init.run()

            for epch in range(n_epochs):
                epoch_time_start = time.time()

                _, rloss = sess.run([opt_step, rnn_loss], feed_dict={x: x_data, y: y_data})

                avg_loss = np.mean(rloss)
                epoch_losses[run_count, epch] = avg_loss

                e_time = time.time() - epoch_time_start
                epoch_times[run_count, epch] = e_time

                print(f"Scope: {var_scope}  Epoch: {epch}   Loss: {avg_loss}   Time: {e_time}")

            sess.close()

        plot_speed_info.append({'data': epoch_times[np.newaxis, run_count, :], 'label': cnfg['label'], 'linewidth': 6.})

        lw = 6. if run_count < 2 else 2.
        plot_loss_info.append({'data': epoch_losses[np.newaxis, run_count, :], 'label': cnfg['label'], 'linewidth': lw})

    pickle.dump(epoch_times, open('epoch_times.p', 'wb'))
    pickle.dump(epoch_losses, open('epoch_losses.p', 'wb'))

    line_plot(plot_speed_info, save_name='gpu_speed_plot.png', n_epochs=n_epochs, ylabel='Epoch time (seconds)')
    line_plot(plot_loss_info, save_name='gpu_loss_plot.png', n_epochs=n_epochs, ylabel='Loss by epoch')

    print('\nEpoch times:')
    print(epoch_times)

    print('\nLosses by epoch:')
    print(epoch_losses)


if __name__ == '__main__':
    main()
