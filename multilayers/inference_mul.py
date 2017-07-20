import tensorflow as tf
import numpy as np
import random
import pickle
import os
import sys
import time

from dataset import *


usage = "python xxx.py saver_model_filename scaler GPU_id"
if len(sys.argv) != 4:
    print("usage:%s" % usage)
    sys.exit()
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[3]

class MLP:
    def __init__(self,saver_model_filename,scaler_filename):
        self.saver_model_filename = saver_model_filename
        self.scaler = self.get_scaler(scaler_filename)
        self.data = dataset(mat_files={"train":["data4/train_zoomout_0.mat"],"test":["inference_result/zoomout_test_1_200.mat"]},scaler=self.scaler)

        #tf graph input 
        self.input_x = tf.placeholder(shape=[None,12416],dtype=tf.float32)
        self.output_y = tf.placeholder(shape=[None,21],dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        #Network parameters 
        self.n_input = 12416
        self.n_hidden_1 = 1024
        self.n_hidden_2 = 1024
        self.num_class = 21 

        #Store layer weights and biases 
        self.weights = {
            'W1': tf.Variable(tf.random_normal([self.n_input,self.n_hidden_1])),
            'W2': tf.Variable(tf.random_normal([self.n_hidden_1,self.n_hidden_2])),
            'W3': tf.Variable(tf.random_normal([self.n_hidden_2,self.num_class]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([self.num_class]))

        }

    def get_scaler(self,scaler_filename):
        f = open(scaler_filename,"rb")
        scaler = pickle.load(f)
        f.close()
        return scaler

    def layer(self,input,weights,biases,batch_norm=True, dropout_prob = 1.0,name=None):
        #layer with batch norm and optional dropout
        W = weights
        b = biases 
        layer = tf.add(tf.matmul(input,W),b) # shape [a,b]
        if batch_norm is True:
            layer = self.batch_norm_layer(layer)
        layer = tf.nn.relu(layer)
        layer = tf.nn.dropout(layer,keep_prob = dropout_prob)
        return layer 
        
    def batch_norm_layer(self,x):
        beta = tf.Variable(tf.zeros(x.shape[1]))
        gamma = tf.Variable(tf.ones(x.shape[1]))
        batch_mean, batch_var = tf.nn.moments(x,axes=[0])
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_update():
            ema_apply_op = ema.apply([batch_mean,batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean),tf.identity(batch_var)
        def mean_update():
            ema_apply_op = ema.apply([batch_mean])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean)
        print("mean: %s" % str(batch_mean))
        #print("mean initialized value:%s" % str(batch_mean.initialized_value))
        print("var: %s" % str(batch_var))
        mean, var = tf.cond(self.is_training, mean_var_update, lambda: (ema.average(batch_mean),ema.average(batch_var)))
        #mean = tf.cond(self.is_training, mean_update, lambda: (ema.average(batch_mean)))
        #var = None
        normed = tf.nn.batch_normalization(x,mean,var,beta,gamma,1e-5)
        return normed

    def multilayer_perceptron(self,x):
        # Layer 1
        self.fc_1 = self.layer(x,self.weights['W1'],self.biases['b1'],dropout_prob = 1.0,name="fc1",batch_norm=True)

        #Layer 2 
        self.fc_2 = self.layer(self.fc_1,self.weights['W2'],self.biases['b2'],batch_norm=True,dropout_prob = self.dropout_prob,name="fc2")

        #Layer 3 
        self.fc_3 = self.layer(self.fc_2,self.weights['W3'],self.biases['b3'],batch_norm = False,dropout_prob = 1.0,name="fc3")

        #softmax layer 
        self.pred = tf.nn.softmax(self.fc_3)

        return self.pred


    def test(self):
        self.pred = self.multilayer_perceptron(self.input_x)
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,self.saver_model_filename)

        gt = None
        pred = None
        while True:
            batch_x, batch_y = self.data.next_batch(batch_num = 1000,category="test")
            epoch = self.data.get_cur_epoch(category="test")
            if epoch > 0: break
            feed_dict = {self.input_x:batch_x,self.is_training:False,self.dropout_prob:1}
            output_ = sess.run(self.pred,feed_dict=feed_dict)
            batch_y = np.argmax(batch_y,axis=1)
            print("batch gt:%s" % batch_y)
            if gt is None: gt = batch_y
            else: gt = np.concatenate([gt,batch_y])
            output_ = np.argmax(output_,axis=1)
            print("output:%s" % output_)
            if pred is None: pred = output_
            else: pred = np.concatenate([pred,output_])
        
        mat_data = {"gt":gt,"pred":pred}
        io.savemat("inference.mat", mat_data)

if __name__ == '__main__':
    MLP = MLP(sys.argv[1],sys.argv[2])
    MLP.test()
