# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:26:14 2018

@author: William
"""

import numpy as np
import tensorflow as tf
#import pickle
#import os


def load_data ():
    data = np.load("cifar_reshape.npy")
    imgs = np.reshape(data, (50000,32,32,3))
    print("loaded")
    return np.float32(imgs)

"""
Generalized convolutional layer
"""
def clayer (i, w, b):
    conv = tf.nn.conv2d(i, w, strides=[1, 2, 2, 1], padding='SAME')
#    output = tf.nn.relu(conv + b)
    output = conv + b
    return output

"""
Generalized deconvolutional layer
"""
def dlayer(j, o):
    deconv = tf.nn.conv2d_transpose(j, weights(o[-1], j.get_shape()[-1].value), output_shape=o, strides=[1,2,2,1])
    output = deconv + bias(o[-1])
    return output

"""
Max pooling will be the same everywhere
"""
#def max_pool (i):
#    output = tf.nn.max_pool(i, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#    return output
"""
Weight initialization
"""
def weights (infeatures, outfeatures):
    return tf.Variable(tf.random_normal(shape = [5,5, infeatures, outfeatures] , stddev=.01))
"""
Biases
"""
def bias (sh):
    return tf.Variable(tf.zeros(shape=sh))

def dense(x, infeatures, outfeatures):
    mat = tf.Variable(tf.random_normal(shape = [infeatures, outfeatures], stddev = .01))
    b = bias (outfeatures)
    output = tf.matmul(x, mat) +b
    return output
    
    
def encoder (x):
    l1 = tf.nn.relu(clayer(x, weights(3,32),bias(32)))
    l2 = tf.nn.relu(clayer(l1, weights(32,64),bias(64)))
    re = tf.reshape(l2, [100, 8*8*64])
    l3 = dense(re, 8*8*64, 20)
    return l3
    
def decoder (x):
    dev = dense(x, 20, 8*8*64)
    
    matrix = tf.nn.relu(tf.reshape(dev, [100, 8, 8, 64]))

    l1 = tf.nn.relu(dlayer(matrix, [100, 16, 16, 32]))    
    l2 = tf.nn.sigmoid(dlayer(l1, [100, 32, 32, 3]))
    return l2



"""
Batching the dataset and making it iterable
"""
#data = load_data()
#print(data)
#batched_data = tf.train.batch(data, batch_size=100, enqueue_many=True)
##print (batched_data)

dataset = tf.data.Dataset.from_tensor_slices(load_data())
batched = dataset.batch(100)
iterator = batched.make_one_shot_iterator()
next_batch = iterator.get_next()


latent_vector = encoder(next_batch)
print("coded")
decompressed = decoder(latent_vector)


"""
Evaluate and minimize compression losses
"""
loss = tf.reduce_mean(tf.square(tf.subtract(next_batch, decompressed)))
print("woo woo")
optimizer = tf.train.AdamOptimizer(.01).minimize(loss)
#saver=tf.train.Saver(max_to_keep = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("yah")
    for i in range(100):
        sess.run((optimizer, loss))
        print("guh")
        print(sess.run(loss))
#        sess.run(latent_vector, feed_dict = {x_: batched_data})




