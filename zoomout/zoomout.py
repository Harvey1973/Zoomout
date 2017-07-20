import scipy.io as io
import zoomout_vgg16
import dataset
import tensorflow as tf

import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


sp_num = 300
batch_size = 10
max_train_data_per_mat = 30
downsample = 2
img_w = 224
img_h = 224
total_f_num = 12416

zo = zoomout_vgg16.Zoomout_Vgg16("zoomout/vgg16.npy",weight=img_w,height=img_h,downsample=downsample)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
data = dataset.data(main_path="data",sp_num=sp_num,weight=img_w,height=img_h,downsample=downsample)

mat_data = {"train_x":[],"train_y":[],"train_slic":[]}

start_time = time.time()
# train data
overflow = False
s = 0
index = 0
while overflow is False:
    print("train %d" % s)
    print("time:%d" % (time.time()-start_time))
    s += 1
    train_data,overflow = data.next_batch(batch=batch_size,kind="train")
    print("1")
    #print("y:%s" % train_data[1])
    zoomout_f = zo.get_zoomout_features(sess,train_data[0],train_data[2])
    mat_data["train_slic"].extend(train_data[2])
    print("2")
    for i in range(len(train_data[1])):
        for j in range(len(train_data[1][i])):
            mat_data["train_x"].append(zoomout_f[i][j])
            tmp = [0] * 21
            #print("train_data[1][j]:%s" % str(train_data[1][j]))
            tmp[train_data[1][i][j]] = 1
            mat_data["train_y"].append(tmp)
    index += batch_size
    if index >= max_train_data_per_mat:
        io.savemat("zoomout_%d_%d.mat" % (s*batch_size - index + 1,s*batch_size),mat_data)
        mat_data = {"train_x":[],"train_y":[],"train_slic":[]}
        index = 0


if len(mat_data["train_x"]) > 0:
    io.savemat("zoomout_%d_%d.mat" % (s*batch_size - index + 1,s*batch_size),mat_data)
# test data
mat_data = {"test_x":[],"test_y":[],"test_slic":[]}
overflow = False
s = 0
while overflow is False:
    print("test %d" % s)
    s += 1
    test_data,overflow = data.next_batch(batch=batch_size,kind="test")
    zoomout_f = zo.get_zoomout_features(sess,test_data[0],test_data[2])
    mat_data["test_slic"].extend(test_data[2])
    for i in range(len(test_data[1])):
        for j in range(len(test_data[1][i])):
            mat_data["test_x"].append(zoomout_f[i][j])
            tmp = [0] * 21
            tmp[test_data[1][i][j]] = 1
            mat_data["test_y"].append(tmp)

io.savemat("zoomout_test.mat",mat_data)
