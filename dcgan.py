#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:52:40 2019

@author: clytie
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from base import Base
import math
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(message)s')

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
def sample_z(size, z_dim):
    return np.random.normal(0, 1, size=[size, z_dim])


class DCGAN(Base):
    def __init__(self, img_dim,
                 z_dim=100,
                 lr=2e-4,
                 max_grad_norm=5,
                 save_model_freq=100,
                 generator_image_freq=100,
                 save_path="./dcgan"):
        assert len(img_dim) == 3
        self.img_dim = img_dim
        self.z_dim = z_dim
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.save_model_freq = save_model_freq
        self.generator_image_freq = generator_image_freq
        
        super().__init__(save_path=save_path)
    
    def _build_network(self):
        self.image = tf.placeholder(
                tf.uint8, [None, *self.img_dim], name="real_inputs")
        self.z = tf.placeholder(
                tf.float32, [None, self.z_dim], name="z")
        
        height, width, channel = self.img_dim
        height //= 16
        width //= 16
        with tf.variable_scope("generator"):
            g = tcl.fully_connected(
                    self.z, height * width * 1024,
                    activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, (-1, height, width, 1024))
            
            g = tcl.conv2d_transpose(
                    g, 512, 3, stride=2, activation_fn=lrelu,
                    normalizer_fn=tcl.batch_norm, padding="SAME",
                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            g = tcl.conv2d_transpose(
                    g, 256, 3, stride=2, activation_fn=lrelu,
                    normalizer_fn=tcl.batch_norm, padding="SAME",
                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            g = tcl.conv2d_transpose(
                    g, 128, 3, stride=2, activation_fn=lrelu,
                    normalizer_fn=tcl.batch_norm, padding="SAME",
                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            
            g = tcl.conv2d_transpose(
                    g, channel, 3, stride=2, activation_fn=tf.nn.tanh, padding="SAME",
                    weights_initializer=tf.random_normal_initializer(0, 0.02))
        
        self.G_sample = g
            
        def __discriminator(img):
            img = tf.subtract(tf.divide(tf.cast(img, tf.float32), 255.0 / 2.0), 1.0)
            shared = tcl.conv2d(
                    img, num_outputs=64, kernel_size=4,
                    stride=2, activation_fn=lrelu)
            
            shared = tcl.conv2d(
                    shared, num_outputs=128, kernel_size=4,
                    stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            
            shared = tcl.conv2d(
                    shared, num_outputs=256, kernel_size=4,
                    stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            
            shared = tcl.conv2d(
                    shared, num_outputs=512, kernel_size=4,
                    stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            
            shared = tcl.flatten(shared)
            
            d = tcl.fully_connected(shared, 1, activation_fn=None, 
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 2, activation_fn=None)
            
            return d, q
        
        with tf.variable_scope("discriminator"):
            self.D_real, _ = __discriminator(self.image)
            
        with tf.variable_scope("discriminator", reuse=True):
            self.D_fake, _ = __discriminator(g)
        
    def _build_algorithm(self):
        G_optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-5)
        D_optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-5)
        
        D_trainable_variables = tf.trainable_variables("discriminator")
        G_trainable_variables = tf.trainable_variables("generator")
        real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)))
        fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.D_loss = real_loss + fake_loss
        
        self.G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
    
        # clip gradients
        D_grads = tf.gradients(self.D_loss, D_trainable_variables)
        G_grads = tf.gradients(self.G_loss, G_trainable_variables)
        
        D_clipped_grads, _ = tf.clip_by_global_norm(D_grads, self.max_grad_norm)
        G_clipped_grads, _ = tf.clip_by_global_norm(G_grads, self.max_grad_norm)
        
        self.D_solver = D_optimizer.apply_gradients(
                zip(D_clipped_grads, D_trainable_variables), global_step=tf.train.get_global_step())
        
        self.G_solver = G_optimizer.apply_gradients(
                zip(G_clipped_grads, G_trainable_variables), global_step=tf.train.get_global_step())
    
    def _generator(self, datas, batch_size):
        n_sample = len(datas)
        index = np.arange(n_sample)
        np.random.shuffle(index)
        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield datas[span_index]
    
    def generate_image(self, epoch):
        save_dir = os.path.join(self.save_path, "images")
        os.makedirs(f"{save_dir}/epoch{epoch}", exist_ok=True)
        G_samples = self.sess.run(
                self.G_sample,
                feed_dict={self.z: sample_z(16, self.z_dim)})
        for i, sample in enumerate(G_samples):
            cv2.imwrite(f"{save_dir}/epoch{epoch}/image{i}.png", (sample + 1) / 2.0 * 255)
    
    def train(self, datas, training_epoches=int(1e6), batch_size=32, G_updates=1):
        assert G_updates > 0
        epoch = 1
        while True:
            data_generator = self._generator(datas, batch_size)
            while True:
                try:
                    img_batch = next(data_generator)
                    cur_batch_size = len(img_batch)
                    
                    _, D_loss_batch = self.sess.run(
                            [self.D_solver, self.D_loss],
                            feed_dict={self.image: img_batch,
                                       self.z: sample_z(cur_batch_size, self.z_dim)}
                            )
                    
                    G_loss_batch_total = 0
                    for _ in range(G_updates):
                        _, G_loss_batch = self.sess.run(
                                [self.G_solver, self.G_loss],
                                feed_dict={self.z: sample_z(cur_batch_size, self.z_dim)})
                        G_loss_batch_total += G_loss_batch
                    logging.info(
                        f">>>>epoch{epoch} D_loss: {D_loss_batch} G_loss: {G_loss_batch_total / G_updates}")
                    
                    if epoch % self.save_model_freq == 0:
                        self.save_model()
                    
                    if epoch % self.generator_image_freq == 0:
                        self.generate_image(epoch)
                    
                    epoch += 1
                    
                except StopIteration:
                    del data_generator
                    break
            
            if epoch > training_epoches:
                break
        
       