import tensorflow as tf
import numpy as np
import random
import os
import sys
import time

from dataset import dataset

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

g = {"l2":0}

def create_net(x,y,batch_num=10,f_weights=None):
    global g
    w_initializer = tf.random_normal_initializer(mean=1.0,stddev=1.0)
    # layer one
    w1 = tf.get_variable("w1",shape=[12416,1024],dtype=tf.float32,initializer=w_initializer)
    b1 = tf.get_variable("b1",shape=[1024],dtype=tf.float32)
    fc1_in = tf.matmul(x,w1)+b1
    fc1_in = tf.contrib.layers.batch_norm(fc1_in,center=True,scale=True,is_training=True)
    fc1 = tf.nn.relu(fc1_in)
    g["l2"] = tf.nn.l2_loss(w1)
    #drop1 = tf.nn.dropout(fc1,keep_prob=0.5)
    # layer two
    w2 = tf.get_variable("w2",shape=[1024,1024],dtype=tf.float32,initializer=w_initializer)
    b2 = tf.get_variable("b2",shape=[1024],dtype=tf.float32)
    #fc2 = tf.nn.relu(tf.matmul(drop1,w2)+b2)
    fc2_in = tf.matmul(fc1,w2)+b2
    fc2_in = tf.contrib.layers.batch_norm(fc2_in,center=True,scale=True,is_training=True)
    fc2 = tf.nn.relu(fc2_in)
    drop2 = tf.nn.dropout(fc2,keep_prob=0.5)
    g["l2"] += tf.nn.l2_loss(w2)
    # layer three
    w3 = tf.get_variable("w3",shape=[1024,21],dtype=tf.float32,initializer=w_initializer)
    b3 = tf.get_variable("b3",shape=[21],dtype=tf.float32)
    fc3_in = tf.matmul(drop2,w3)+b3
    fc3_in = tf.contrib.layers.batch_norm(fc3_in,center=True,scale=True,is_training=True)
    fc3 = tf.nn.relu(fc3_in)
    g["l2"] += tf.nn.l2_loss(w3)
    # the softmax cross entropy
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3,labels=y))
    #loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y,fc3,f_weights))
    pred = tf.nn.softmax(fc3)
    pred_ = tf.log(pred+1e-10)
    tmp = f_weights * y
    loss = -tf.reduce_mean(pred_*tmp)

    #pred_label = tf.argmax(fc3,1)
    y_label = tf.argmax(y,1)
    y_bg_label = np.zeros([batch_num])
    nbg_tmp = tf.logical_not(tf.equal(y_label,y_bg_label))
    bg_count = batch_num - tf.reduce_sum(tf.cast(nbg_tmp,dtype=tf.float32))

    pred_label = tf.argmax(pred*f_weights,1)
    all_tmp = tf.equal(pred_label,y_label)
    accu_tmp = tf.logical_and(all_tmp,nbg_tmp)
    accuracy = tf.reduce_mean(tf.cast(accu_tmp,dtype=tf.float32))

    pred_label = tf.argmax(pred,1)
    all_tmp = tf.equal(pred_label,y_label)
    accu_tmp = tf.logical_and(all_tmp,nbg_tmp)
    accuracy_2 = tf.reduce_mean(tf.cast(accu_tmp,dtype=tf.float32))


    pred_label = tf.argmax(pred*f_weights,1)
    all_tmp = tf.equal(pred_label,y_label)
    accuracy_bg = tf.reduce_mean(tf.cast(all_tmp,dtype=tf.float32))

    pred_label = tf.argmax(pred,1)
    all_tmp = tf.equal(pred_label,y_label)
    accuracy_2_bg = tf.reduce_mean(tf.cast(all_tmp,dtype=tf.float32))
    return loss,accuracy,accuracy_2,accuracy_bg,accuracy_2_bg,bg_count,pred_label,pred

def main():
    start_time = time.time()
    global g
    batch_num = 100
    num_of_classes = 21
    lr_rate = 0.001
    weight_decay = 0.001
    data = dataset()
    data.decimate_bg_sp(0.5)
    histogram = data.get_histogram()
    f_weights = []
    all_count = 0
    for one in range(num_of_classes):
        key = str(one)
        f_weights.append(histogram[key])
        all_count += histogram[key]
    f_weights = list(map(lambda x: all_count/x,f_weights))
    print("f_weights:%s" % str(f_weights))
    f_weights = np.array(f_weights)

    input_x = tf.placeholder(shape=[None,12416],dtype=tf.float32)
    output_y = tf.placeholder(shape=[None,21],dtype=tf.float32)
    loss,accu,accu2,accu_bg,accu2_bg,bg_count,pred_label,pred = create_net(input_x,output_y,batch_num,f_weights)
    lr = tf.Variable(lr_rate,trainable=False)
    lr_update = tf.assign(lr,lr*(1-weight_decay))
    opt = tf.train.GradientDescentOptimizer(learning_rate = lr)
    #opt = tf.train.GradientDescentOptimizer(learning_rate = lr_rate)
    #train_op = opt.minimize(loss+g["l2"])
    gradient = opt.compute_gradients(loss+g["l2"])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(gradient)

    # add summary
    tf.summary.scalar("loss",loss)
    tf.summary.scalar("l2",g["l2"])
    tf.summary.scalar("loss+l2",loss+g["l2"])
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("logs")
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epoch = data.get_cur_epoch()
    old_epoch = epoch
    
    batch = 0
    time_tmp = time.time()
    print("before training: %d" % (time_tmp - start_time))
    while epoch <= 300:
        batch_x,batch_y = data.next_batch(batch_num = batch_num)
        #batch_y = np.argmax(batch_y,axis=1)
        feed_dict_tmp = {input_x:batch_x, output_y:batch_y}
        _ = sess.run(train_op,feed_dict=feed_dict_tmp)

        if batch % 100 == 99:
            loss_,accu_,accu2_,accu_bg_,accu2_bg_,bg_count_,summary_,lr_ = sess.run([loss,accu,accu2,accu_bg,accu2_bg,bg_count,summary_op,lr],feed_dict=feed_dict_tmp)
            time_tmp = time.time()
            print("time:%d,epoch:%d,batch:%d,loss:%f,accuracy:%f,accuracy_bg:%f,accuracy2:%f,accuracy2_bg:%f,bg_count:%d,lr:%f" % ((time_tmp-start_time),epoch,batch,loss_,accu_,accu_bg_,accu2_,accu2_bg_,bg_count_,lr_))
            summary_writer.add_summary(summary_,batch)

        if batch % 1000 == 999:
            grad_,pred_,pred_label_ = sess.run([gradient,pred,pred_label],feed_dict=feed_dict_tmp)
            #print("grad_:%s" % str(grad_))
            print("pred_:%s" % str(pred_[0]))
            index_ = random.choice(range(1,batch_num))
            print("pred_:%s" % str(pred_[index_]))
            print("pred label:%s" % str(pred_label_))
            print("gt   label:%s" % str(np.argmax(batch_y,1)))

            batch_x,batch_y = data.next_batch(batch_num = batch_num,category="test")
            feed_dict_tmp = {input_x:batch_x, output_y:batch_y}
            loss_,accu_,accu2_,accu_bg_,accu2_bg_,bg_count_,summary_,lr_ = sess.run([loss,accu,accu2,accu_bg,accu2_bg,bg_count,summary_op,lr],feed_dict=feed_dict_tmp)
            time_tmp = time.time()
            print("test time:%d,epoch:%d,batch:%d,loss:%f,accuracy:%f,accuracy_bg:%f,accuracy2:%f,accuracy2_bg:%f,bg_count:%d,lr:%f" % ((time_tmp-start_time),epoch,batch,loss_,accu_,accu_bg_,accu2_,accu2_bg_,bg_count_,lr_))
        batch += 1
        epoch = data.get_cur_epoch()
        #if epoch != old_epoch:
            #sess.run(lr_update)
        old_epoch = epoch

if __name__ == "__main__":
    main()
