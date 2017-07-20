import tensorflow as tf
import numpy as np
import random
import os
import sys
import time
import matplotlib.pyplot as plt

from dataset import *
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

class MLP:
    def __init__(self):
        self.learning_rate = 0.001
        self.learning_rate_decay = 0.1
        self.weight_decay = 0.1
        self.batch_num = 100
        self.batch_num_t = 1000
        self.data = dataset()
        #self.data.decimate_bg_sp(0.97)
        #self.data.decimate_bg_sp(0,category="test")
        self.f_weights_v = []
        self.count = 0
        self.num_class = 21
        self.histogram = self.data.get_histogram()
        print("histogram: %s" % str(self.histogram))
        self.g = {'L2':0} 
        for one in range(self.num_class):
            key = str(one)
            self.f_weights_v.append(self.histogram[key])
            self.count +=self.histogram[key]
        self.f_weights_v = list(map(lambda x: self.count/x,self.f_weights_v))
        print("weights: %s" % str(self.f_weights_v))
        self.f_weights_v = np.array(self.f_weights_v)
        #tf graph input 
        self.input_x = tf.placeholder(shape=[None,12416],dtype=tf.float32)
        self.output_y = tf.placeholder(shape=[None,21],dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)
        #self.is_training = True
        #Network parameters 
        self.n_input = 12416
        self.n_hidden_1 = 8192
        self.n_hidden_2 = 4096
        self.n_hidden_3 = 2048
        self.n_hidden_4 = 1024
        self.n_hidden_5 = 1024
        self.num_class = 21 
        #Store layer weights and biases 
        self.weights = {
            'W1': tf.Variable(tf.random_normal([self.n_input,self.n_hidden_1])),
            'W2': tf.Variable(tf.random_normal([self.n_hidden_1,self.n_hidden_2])),
            'W3': tf.Variable(tf.random_normal([self.n_hidden_2,self.n_hidden_3])),
            'W4': tf.Variable(tf.random_normal([self.n_hidden_3,self.n_hidden_4])),
            'W5': tf.Variable(tf.random_normal([self.n_hidden_4,self.n_hidden_5])),
            'W6': tf.Variable(tf.random_normal([self.n_hidden_5,self.num_class]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'b4': tf.Variable(tf.random_normal([self.n_hidden_4])),
            'b5': tf.Variable(tf.random_normal([self.n_hidden_5])),
            'b6': tf.Variable(tf.random_normal([self.num_class]))

        }

    def layer(self,input,weights,biases,batch_norm=True, dropout_prob = 1.0,name=None):
        #layer with batch norm and optional dropout
        W = weights
        b = biases 
        layer = tf.add(tf.matmul(input,W),b) # shape [a,b]
        if batch_norm is True:
            #layer = self.batch_norm_layer(layer)
            layer = tf.layers.batch_normalization(layer,center=True,scale=True)
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

    def multilayer_perceptron(self,x,y,batch_num,batch_num_t,f_weights_v):
        # Layer 1
        self.fc_1 = self.layer(x,self.weights['W1'],self.biases['b1'],dropout_prob = 1.0,name="fc1",batch_norm=True)
        #store weights 
        self.g["L2_layer1"] = tf.nn.l2_loss(self.weights['W1'])
        self.g['L2'] += self.g["L2_layer1"]

        #Layer 2 
        self.fc_2 = self.layer(self.fc_1,self.weights['W2'],self.biases['b2'],batch_norm=True,dropout_prob = self.dropout_prob,name="fc2")
        self.g['L2_layer2'] = tf.nn.l2_loss(self.weights['W2'])
        self.g['L2'] += self.g["L2_layer2"]


        #Layer 3 

        self.fc_3 = self.layer(self.fc_2,self.weights['W3'],self.biases['b3'],batch_norm = False,dropout_prob = 1.0,name="fc3")
        self.g['L2_layer3']= tf.nn.l2_loss(self.weights['W3'])
        self.g['L2'] += self.g["L2_layer3"]



        #Layer 4 
        self.fc_4 = self.layer(self.fc_3,self.weights['W4'],self.biases['b4'],batch_norm = True,dropout = True,name="fc4")
        self.g['L2']+= tf.nn.l2_loss(self.weights['W4'])
        
        #Layer 5 
        self.fc_5 = self.layer(self.fc_4,self.weights['W5'],self.biases['b5'],batch_norm = True,dropout = True,name="fc5")
        self.g['L2']+= tf.nn.l2_loss(self.weights['W5'])
        
        #Layer 6 
        self.fc_6 = self.layer(self.fc_5,self.weights['W6'],self.biases['b6'],batch_norm = False,dropout = True,name="fc6")
        self.g['L2']+= tf.nn.l2_loss(self.weights['W6'])

        #softmax layer 

        self.pred = tf.log(tf.nn.softmax(self.fc_3)+1e-10)
        self.inverse_freq = f_weights_v*y

        self.loss = -tf.reduce_mean(self.pred*self.inverse_freq)

        #accuracy(without backgroud)
        self.pred_label = tf.argmax(self.pred,1)
        self.y_label = tf.argmax(y,1)
        self.y_bg_label = np.zeros([batch_num])
        self.y_bg_label_t = np.zeros([batch_num_t])
        self.nbg = tf.logical_not(tf.equal(self.y_label,self.y_bg_label)) #only the non-background pixel will have value True
        self.nbg_t = tf.logical_not(tf.equal(self.y_label,self.y_bg_label_t)) #only the non-background pixel will have value True
        
        all_tmp = tf.equal(self.pred_label,self.y_label)
        accu_tmp = tf.logical_and(all_tmp,self.nbg)
        accu_tmp_t = tf.logical_and(all_tmp,self.nbg_t)

        TP = tf.reduce_mean(tf.cast(accu_tmp,dtype = tf.float32))
        TP_TN = tf.reduce_mean(tf.cast(all_tmp,dtype=tf.float32))
        TP_FN = tf.reduce_mean(tf.cast(self.nbg,dtype = tf.float32))
        self.accuracy = TP / TP_FN
        self.accuracy_bg = TP_TN
        TP_t = tf.reduce_mean(tf.cast(accu_tmp_t,dtype = tf.float32))
        TP_TN_t = tf.reduce_mean(tf.cast(all_tmp,dtype=tf.float32))
        TP_FN_t = tf.reduce_mean(tf.cast(self.nbg_t,dtype = tf.float32))
        self.accuracy_t = TP_t / TP_FN_t
        self.accuracy_bg_t = TP_TN_t

        return self.loss,self.accuracy,self.accuracy_bg,self.accuracy_t,self.accuracy_bg_t,self.pred_label,self.pred


    def train(self):
        start_time = time.time()
        loss,accuracy,accuracy_bg,accuracy_t,accuracy_bg_t,pred_label,pred = self.multilayer_perceptron(self.input_x,self.output_y,self.batch_num,self.batch_num_t,self.f_weights_v)
        lr = tf.Variable(self.learning_rate,trainable=False)

        lr_update = tf.assign(lr,lr*self.learning_rate_decay)
        #opt = tf.train.GradientDescentOptimizer(learning_rate = lr)
        opt = tf.train.AdamOptimizer(learning_rate = lr)
        gradient = opt.compute_gradients(self.loss+self.weight_decay*self.g["L2"])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #adds additional ops on the gragh for updating moving average and variance 
        with tf.control_dependencies(update_ops):               #adds update ops as dependencies of the training op 
            #train_op = optimizer.apply_gradients(gradient)
            train_op = optimizer.minimize(self.loss+self.g['L2']) 
        # add summary
        tf.summary.scalar("loss",self.loss)
        tf.summary.scalar("l2",self.g["L2"])
        tf.summary.scalar("l2_layer1",self.g["L2_layer1"])
        tf.summary.scalar("l2_layer2",self.g["L2_layer2"])
        tf.summary.scalar("l2_layer3",self.g["L2_layer3"])
        tf.summary.scalar("loss+l2",self.loss+self.g["L2"])
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("logs")
        
        tf.add_to_collection("input",self.input_x)
        tf.add_to_collection("is_training",self.is_training)
        tf.add_to_collection("dropout_prob",self.dropout_prob)
        tf.add_to_collection("output",self.output_y)
        saver = tf.train.Saver()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        # store losses 
        #losses = []
        f = open('train_loss.txt','w')
        f_2 = open('test_loss.txt','w')
        epoch = self.data.get_cur_epoch()
        old_epoch = epoch
        batch = 0
        time_tmp = time.time()

        while epoch <= 50:

            batch_x,batch_y = self.data.next_batch(batch_num = self.batch_num)
            #batch_y = np.argmax(batch_y,axis=1)
            feed_dict_tmp = {self.input_x:batch_x, self.output_y:batch_y,self.is_training:True,self.dropout_prob:0.5}
            #feed_dict_tmp = {self.input_x:batch_x, self.output_y:batch_y}
            _ = sess.run(train_op,feed_dict=feed_dict_tmp)

            if batch % 100 == 99:
                loss_,accu_,accu_bg_,summary_,lr_,loss_l2_,l2_layer1_,l2_layer2_,l2_layer3_ = sess.run([loss,accuracy,accuracy_bg,summary_op,lr,self.g["L2"],self.g["L2_layer1"],self.g["L2_layer2"],self.g["L2_layer3"]],feed_dict=feed_dict_tmp)
                time_tmp = time.time()
                print("time:%d,epoch:%d,batch:%d,loss:%f,loss_l2:%f,l2_layer1:%f,l2_layer2:%f,l2_layer3:%f,accuracy:%f,accuracy_bg:%f,lr:%g" % ((time_tmp-start_time),epoch,batch,loss_,loss_l2_,l2_layer1_,l2_layer2_,l2_layer3_,accu_,accu_bg_,lr_))
                summary_writer.add_summary(summary_,batch)

            if batch % 1000 == 999:
                #grad_,pred_,pred_label_ = sess.run([gradient,pred,pred_label],feed_dict=feed_dict_tmp)
                pred_,pred_label_ = sess.run([pred,pred_label],feed_dict=feed_dict_tmp)
                #print("grad_:%s" % str(grad_))
                #print("pred_:%s" % str(pred_[0]))
                index_ = random.choice(range(1,self.batch_num))
                print("pred_:%s" % str(pred_[index_]))
                print("pred label:%s" % str(pred_label_))
                print("gt   label:%s" % str(np.argmax(batch_y,1)))

            if batch % 1000 == 999:
                batch_x,batch_y = self.data.next_batch(batch_num = self.batch_num_t,category="test")
                #batch_x,batch_y = self.data.next_batch_for_test()
                feed_dict_tmp = {self.input_x:batch_x, self.output_y:batch_y,self.is_training:False,self.dropout_prob:1.0}
                loss_,accu_,accu_bg_,summary_,lr_ = sess.run([loss,accuracy_t,accuracy_bg_t,summary_op,lr],feed_dict=feed_dict_tmp)
                time_tmp = time.time()
                #print("test time:%d,epoch:%d,batch:%d,loss:%f,accuracy:%f,lr:%g" % ((time_tmp-start_time),epoch,batch,loss_,accu_,lr_))
                print("test time:%d,epoch:%d,batch:%d,loss:%f,accuracy:%f,accuracy_bg:%f,lr:%g" % ((time_tmp-start_time),epoch,batch,loss_,accu_,accu_bg_,lr_))
                grad_,pred_,pred_label_ = sess.run([gradient,pred,pred_label],feed_dict=feed_dict_tmp)
                #print("grad_:%s" % str(grad_))
                #print("pred_:%s" % str(pred_[0]))
                index_ = random.choice(range(1,self.batch_num_t))
                print("pred_:%s" % str(pred_[index_]))
                print("pred label:%s" % str(pred_label_))
                print("gt   label:%s" % str(np.argmax(batch_y,1)))
                if accu_ > 0.5:
                    saver.save(sess,"saver/saver_%d_%f" % (epoch,accu_))
            batch += 1
            epoch = self.data.get_cur_epoch()
            if epoch % 25 == 24 and old_epoch != epoch:
                sess.run(lr_update)
            old_epoch = epoch


if __name__ == '__main__':
    MLP = MLP()
    MLP.train()
