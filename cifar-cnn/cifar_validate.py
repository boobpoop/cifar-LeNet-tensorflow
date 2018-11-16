import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward_propagation
import cifar_train
import load_cifar_dataset

def valuate():

    #step 1------define placeholder of input data
    input_data_x = tf.placeholder(tf.float32, [None, forward_propagation.IMAGE_SIZE, forward_propagation.IMAGE_SIZE, forward_propagation.INPUT_CHANNELS], name = "input_data_x")
    input_data_y = tf.placeholder(tf.float32, [None, forward_propagation.OUTPUT_NODE], name = "input_data_y")

    #step 2------use network struct
    #goto forward_progation.py

    #step 3-----calculate forward propagation
    (y, _1, _2, _3) = forward_propagation.forward_propagation(input_data_x)
   
    #step 4------predict accuracy   
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(input_data_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #step 5------define a object to load model
    saver = tf.train.Saver()

    #step 6------execution
    with tf.Session() as sess:
        (data_test, labels_test) = load_cifar_dataset.get_test_data()
        validate_feed = {input_data_x: data_test, input_data_y: labels_test}

        #fine model
        ckpt = tf.train.get_checkpoint_state(cifar_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            #load model
            saver.restore(sess, ckpt.model_checkpoint_path)
            accuracy_prediction = sess.run(accuracy, feed_dict = validate_feed)
            print("accuracy on validation data is %g" %(accuracy_prediction))
        else:
            print("No checkpoint file found")
            return

def main(argc = None):
    valuate()

if __name__ == "__main__":
    tf.app.run()
