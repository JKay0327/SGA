# coding:utf-8
'''
Created on 2020年10月21日
@author: Chyi
'''
#import the modules
#@title Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import numpy as np
import sonnet as snt
import tensorflow as tf 
import kfac

import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import gaussian_kde

#@title Defining the SGA Optimiser

def list_divide_scalar(xs, y):
  return [x / y for x in xs]


def list_subtract(xs, ys):
  return [x - y for (x, y) in zip(xs, ys)]


def jacobian_vec(ys, xs, vs):
    return kfac.utils.fwd_gradients(
        ys, xs, grad_xs=vs, stop_gradients=xs)


def jacobian_transpose_vec(ys, xs, vs):
    dydxs = tf.gradients(ys, xs, grad_ys=vs, stop_gradients=xs)
    dydxs = [
        tf.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)
  ]
    return dydxs


def _dot(x, y):
    dot_list = []
    for xx, yy in zip(x, y):
        dot_list.append(tf.reduce_sum(xx * yy))
    return tf.add_n(dot_list)


class SymplecticOptimizer(tf.train.Optimizer):
    """Optimizer that corrects for rotational components in gradients."""

    def __init__(self,
               learning_rate,
               reg_params=1.,
               use_signs=True,
               use_locking=False,
               name='symplectic_optimizer'):
        super(SymplecticOptimizer, self).__init__(
            use_locking=use_locking, name=name)
        self._gd = tf.train.RMSPropOptimizer(learning_rate)
        self._reg_params = reg_params
        self._use_signs = use_signs

    def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=tf.train.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
        return self._gd.compute_gradients(loss, var_list, gate_gradients,
                                      aggregation_method,
                                      colocate_gradients_with_ops, grad_loss)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        grads, vars_ = zip(*grads_and_vars)
        n = len(vars_)
        h_v = jacobian_vec(grads, vars_, grads)
        ht_v = jacobian_transpose_vec(grads, vars_, grads)
        at_v = list_divide_scalar(list_subtract(ht_v, h_v), 2.)
        if self._use_signs:
            grad_dot_h = _dot(grads, ht_v)
            at_v_dot_h = _dot(at_v, ht_v)
            mult = grad_dot_h * at_v_dot_h
            lambda_ = tf.sign(mult / n + 0.1) * self._reg_params
        else:
            lambda_ = self._reg_params
        apply_vec = [(g + lambda_ * ag, x)
                     for (g, ag, x) in zip(grads, at_v, vars_)
                     if at_v is not None]
        return self._gd.apply_gradients(apply_vec, global_step, name)


#@title An MLP Sonnet module

class MLP(snt.AbstractModule):
#An MLP with hidden layers of the same width as the input.
    def __init__(self, depth, hidden_size, out_dim, name='SimpleNet'):
        super(MLP, self).__init__(name=name)
        self._depth = depth
        self._hidden_size = hidden_size
        self._out_dim = out_dim

    def _build(self, input):
        h = input
        for i in range(self._depth):
            h = tf.nn.relu(snt.Linear(self._hidden_size)(h))
        return snt.Linear(self._out_dim)(h)

 
#@title Build Graph to train a GAN

def reset_and_build_graph(
    depth, width, x_real_builder, z_dim, batch_size, learning_rate, mode):
    tf.reset_default_graph()

    x_real = x_real_builder(batch_size)
    x_dim = x_real.get_shape().as_list()[1]
    generator = MLP(depth, width, x_dim, 'generator')
    discriminator = MLP(depth, width, 1, 'discriminator')
    z = tf.random_normal([batch_size, z_dim])
    x_fake = generator(z)
    disc_out_real = discriminator(x_real)
    disc_out_fake = discriminator(x_fake)

# Loss
    disc_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_real, labels=tf.ones_like(disc_out_real)))
    disc_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
          logits=disc_out_fake, labels=tf.zeros_like(disc_out_fake)))
    disc_loss = disc_loss_real + disc_loss_fake

    gen_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_fake, labels=tf.ones_like(disc_out_fake)))
    gen_vars = generator.variable_scope.trainable_variables()
    disc_vars = discriminator.variable_scope.trainable_variables()
# Compute gradients
    xs = disc_vars + gen_vars  
    disc_grads = tf.gradients(disc_loss, disc_vars)
    gen_grads = tf.gradients(gen_loss, gen_vars)
    Xi = disc_grads + gen_grads
    apply_vec = list(zip(Xi, xs))

    if mode == 'RMS':
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif mode == 'SGA':
        optimizer = SymplecticOptimizer(learning_rate)
    else:
        raise ValueError('Mode %s not recognised' % mode)

    with tf.control_dependencies([g for (g, v) in apply_vec]):
        train_op = optimizer.apply_gradients(apply_vec)

    init = tf.global_variables_initializer()
  
    return train_op, x_fake, z, init, disc_loss, gen_loss

#@title Defining the functions for the experiment

def kde(mu, tau, bbox=None, xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    kernel = gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    # 将生成的图片保存
    # plt.savefig("SGA_figure.png")
    # plt.show()


def train(train_op, x_fake, z, init, disc_loss, gen_loss, z_dim,
          n_iter=10001, n_save=5000):
    bbox = [-2, 2, -2, 2]
    batch_size = x_fake.get_shape()[0].value
    ztest = [np.random.randn(batch_size, z_dim) for i in range(10)]

    with tf.Session() as sess:
        sess.run(init)

        for i in range(n_iter):
            disc_loss_out, gen_loss_out, _ = sess.run(
                [disc_loss, gen_loss, train_op])
            if i % n_save == 0:
                print('i = %d, discriminant loss = %.4f, generator loss =%.4f' %
                      (i, disc_loss_out, gen_loss_out))
                x_out = np.concatenate(
                    [sess.run(x_fake, feed_dict={z: zt}) for zt in ztest], axis=0)
                plt.figure()
                kde(x_out[:, 0], x_out[:, 1], bbox=bbox)
                plt.savefig("SGA_figure.png")
    
def learn_mixture_of_gaussians(mode):
    print(mode)
    def x_real_builder(batch_size):
        sigma = 0.1
        skel = np.array([
            [ 1.50,  1.50],
            [ 1.50,  0.50],
            [ 1.50, -0.50],
            [ 1.50, -1.50],
            [ 0.50,  1.50],
            [ 0.50,  0.50],
            [ 0.50, -0.50],
            [ 0.50, -1.50],
            [-1.50,  1.50],
            [-1.50,  0.50],
            [-1.50, -0.50],
            [-1.50, -1.50],
            [-0.50,  1.50],
            [-0.50,  0.50],
            [-0.50, -0.50],
            [-0.50, -1.50],
        ])
        temp = np.tile(skel, (batch_size // 16 + 1,1))#size=272,2
        mus = temp[0:batch_size,:]
        return mus + sigma*tf.random_normal([batch_size, 2])*.2
  
    z_dim = 64
    train_op, x_fake, z, init, disc_loss, gen_loss = reset_and_build_graph(
        depth=6, width=384, x_real_builder=x_real_builder, z_dim=z_dim,
        batch_size=256, learning_rate=1e-4, mode=mode)

    train(train_op, x_fake, z, init, disc_loss, gen_loss, z_dim)


if __name__=="__main__":
    # Use plain RmsProp to optimise the GAN parameters.
    # This experiment demonstratest mode collapse in traditional GAN training.
#learn_mixture_of_gaussians('RMS')
    # Use Symplectic Gradient Adjustment to optimise the GAN parameters.
    # With SGA, all modes are produced by the trained GAN.
    learn_mixture_of_gaussians('SGA')







