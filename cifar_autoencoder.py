# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:26:14 2018

@author: William

Trains an autoencoder on a preformatted version of the cifr 10 dataset.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
"""
Create plot
"""
plt.ion()
f, arrax = plt.subplots(2,1)

def load_data ():
    data = np.load("cifar_reshape.npy")
    imgs = np.reshape(data, (50000,32,32,3))
    imgs = np.divide(imgs, 255)
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

    mat = tf.Variable(tf.random_normal(shape = [infeatures, outfeatures], stddev = .1))
    b = bias (outfeatures)
    output = tf.matmul(x, mat) +b
    return output
    
    
def encoder (x):
	l1 = tf.nn.relu(clayer(x, weights(3,96), bias(96)))
	l2 = tf.nn.relu(clayer(l1, weights(96,192),bias(192)))
	l3 = tf.nn.relu(clayer(l2, weights(192,288),bias(288)))
	re = tf.reshape(l3, [50, 4*4*288])
	l4 = dense(re, 4*4*288, 600)
	return l4
    
def decoder (x):
	dev = dense(x, 600, 4*4*288)
    
	matrix = (tf.reshape(dev, [50, 4, 4, 288]))
	

	l1 = tf.nn.relu(dlayer(matrix, [50, 8, 8, 192]))  
	l2 = tf.nn.relu(dlayer(l1, [50, 16, 16, 96]))
	l3 = (dlayer(l2, [50, 32, 32, 3]))
	
	norm = tf.nn.relu(l3)
	return l3, norm



"""
Batching the dataset and making it iterable
"""
data = load_data()
dataset = tf.data.Dataset.from_tensor_slices(data)
batched = dataset.batch(50).repeat()
iterator = batched.make_one_shot_iterator()
next_batch = iterator.get_next()

#Old way of batching and interating data for dict_feed
#data = load_data()
#print(data)
#batched_data = tf.train.batch(data, batch_size=100, enqueue_many=True)
##print (batched_data)


latent_vector = encoder(next_batch)
print(latent_vector)
reconstructed, decompressed = decoder(latent_vector)


"""
Evaluate and minimize compression losses
"""
#loss = tf.reduce_mean(tf.square(tf.subtract(next_batch, decompressed)))
loss = tf.reduce_mean(tf.square(tf.subtract( tf.multiply(tf.constant(255,dtype=tf.float32),next_batch), tf.multiply(tf.constant(255,dtype=tf.float32),decompressed)) ))
print("woo woo")
optimizer = tf.train.AdamOptimizer(.001).minimize(loss)

saver=tf.train.Saver(max_to_keep = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("yah")
    for i in range(10000):
        sess.run((loss, optimizer))
        print("iteration " + str(i))

#        Display the original and decompressed image in matplotlib, save the images as npy files, and
#        save the model every 500 iterations
        if (i %500 == 0):
                print("loss at " +str(i)+": "+ str(sess.run(loss)))
                original, decoded = sess.run((next_batch[10],tf.abs(decompressed[10])))
          
                arrax[0].imshow(original)                
                arrax[1].imshow(decoded)
                plt.show()
                plt.pause(30)
                np.save("original"+str(i)+".npy", original)
                np.save("decoded"+str(i)+".npy", decoded)				
                savefile=saver.save(sess, "/tmp/cifar_v3.ckpt")
			
