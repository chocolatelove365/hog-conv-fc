#!/usr/bin/env python3
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import sys

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

def max_pool_5x5(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='SAME')

def calc_performance(pred, true):
    pred_size = pred.shape[0]
    true_size = true.shape[0]
    pred = np.reshape(pred, (pred_size))
    true = np.reshape(true, (true_size))
    TP = np.dot(pred, true)
    FP = np.dot(pred, 1-true)
    FN = np.dot(1-pred, true)
    TN = np.dot(1-pred, 1-true)
    print("TP FP FN TN : ")
    print(TP, FP, FN, TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fmeasure = (2 * recall * precision) / (recall + precision)
    return precision, recall, fmeasure

#### Load data ####
data = np.load("hog_data/train_data.npy")
label = np.load("hog_data/train_label.npy")
test_data = np.load("hog_data/test_data.npy")
test_label = np.load("hog_data/test_label.npy")

#x_image = tf.placeholder(tf.float32, [None, 30, 30, 1])
x_image = tf.placeholder(tf.float32, [None, 6*6*9])
x_image2 = tf.reshape(x_image, [-1, 6, 6, 9])

W_conv = weight_variable([3, 3, 9, 32])
b_conv = bias_variable([32])
h_conv = tf.nn.relu(conv2d(x_image2, W_conv) + b_conv)
h_pool = max_pool_3x3(h_conv)

W_fc = weight_variable([2 * 2 * 32, 1])
b_fc = bias_variable([1])

h_pool_flat = tf.reshape(h_pool, [-1, 2 * 2 * 32])

keep_prob = tf.placeholder(tf.float32)
h_pool_flat_drop = tf.nn.dropout(h_pool_flat, keep_prob)

y_conv = tf.matmul(h_pool_flat_drop, W_fc) + b_fc
y_ = tf.placeholder(tf.float32, [None, 1])

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_conv,y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

y_p = tf.round(tf.sigmoid(y_conv))
correct_prediction = tf.equal(tf.round(tf.sigmoid(y_conv)), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#### Initialize ####
sess = tf.InteractiveSession()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state("./model")
if ckpt:
    last_model = ckpt.model_checkpoint_path
    print("Load model")
    saver.restore(sess, last_model)
else:
    print("Initialize model")
    sess.run(tf.global_variables_initializer())

#### Train ####
num_data = len(data)
num_test_data = len(test_data)
NUM_TRAIN = 1000
N = 20


for i in range(NUM_TRAIN):
    sys.stdout.write("Epoch %d/%d\n" % (i+1, NUM_TRAIN))
    BATCH_SIZE = 400
    idx = np.random.permutation(num_data)
    for j in range(0, num_data, BATCH_SIZE):
        sys.stdout.write("Step %d\n" %(j))
        batch_xs = data[idx][j:j+BATCH_SIZE]
        batch_ys = label[idx][j:j+BATCH_SIZE]
        train_step.run(feed_dict={x_image: batch_xs, y_: batch_ys, keep_prob:0.5})
        step = j / BATCH_SIZE
        if step % N == 0:
            ent, acc, y_pred, y_true = sess.run([cross_entropy, accuracy, y_p, y_], feed_dict={x_image:data, y_:label, keep_prob:1.0})
            print(y_pred)
            print(y_true)
            sys.stdout.write("Step %d\n train cross_entropy: %f, accuracy: %f\n" % (step, ent, acc))
            precision, recall, fmeasure = calc_performance(y_pred, y_true)
            sys.stdout.write("       precision: %f, recall: %f, F-measure: %f\n" % (precision, recall, fmeasure))
            #### Test trained model ####
            test_ent, test_acc, test_y_pred, test_y_true = sess.run([cross_entropy, accuracy, y_p, y_], feed_dict={x_image:test_data, y_:test_label[0:num_test_data,:], keep_prob:1.0})
            sys.stdout.write(" test cross_entropy: %f, accuracy: %f\n" % (test_ent, test_acc))
    
            precision, recall, fmeasure = calc_performance(test_y_pred, test_y_true)
            sys.stdout.write("       precision: %f, recall: %f, F-measure: %f\n" % (precision, recall, fmeasure))

        #### Save trained model ####
        saver.save(sess, "model/model.ckpt")
