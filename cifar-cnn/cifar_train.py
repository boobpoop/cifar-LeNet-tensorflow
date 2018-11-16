import tensorflow as tf
import forward_propagation
import os
import pickle
import pandas as pd
import numpy as np
import load_cifar_dataset

MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "model"
MODEL_NAME = "model.ckpt"
LAMDA = 0.004                        

def train():
    #step 1------define placeholder of input data
    input_data_x = tf.placeholder(tf.float32, [None, forward_propagation.IMAGE_SIZE, forward_propagation.IMAGE_SIZE, forward_propagation.INPUT_CHANNELS], name = "input_data_x")
    input_data_y = tf.placeholder(tf.float32, [None, forward_propagation.OUTPUT_NODE], name = "input_data_y")
    
    #step 2------define network structure
    #goto forward_propagation.py

    #step 3------calculate forward propagation
    (y, weight1, weight2, weight3) = forward_propagation.forward_propagation(input_data_x)
   
    #step 4------define loss
    #cross_entropy = tf.reduce_mean( -tf.reduce_sum(input_data_y * tf.log(tf.clip_by_value(y, 1e-8, tf.reduce_max(y))), reduction_indices = [1]))
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = input_data_y))
    #use l2_regularization
    l2_w1 = LAMDA * tf.nn.l2_loss(weight1)
    l2_w2 = LAMDA * tf.nn.l2_loss(weight2)
    l2_w3 = LAMDA * tf.nn.l2_loss(weight3)
    loss = cross_entropy + l2_w1 + l2_w2 + l2_w3

    #step 5------train
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    #step 6------define a object to save model
    saver = tf.train.Saver()

    #step 8------execution
    TRAINING_STEP = 20000
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        (data_train, labels_train) = load_cifar_dataset.get_train_data("cifar_dataset")
        for steps in range(1, TRAINING_STEP):
            start = steps * forward_propagation.BATCH_SIZE % 50000
            _, loss_value = sess.run([train_step, loss], feed_dict = {input_data_x: data_train[start: start + forward_propagation.BATCH_SIZE], input_data_y: labels_train[start: start + forward_propagation.BATCH_SIZE]})
            if steps % 1000 == 1:
                print("After %d steps, loss on training batch is %g" %(steps, loss_value))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

def main(argv = None):
    train()

if __name__ == "__main__":
    tf.app.run()
    
