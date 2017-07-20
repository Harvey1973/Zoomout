import os

import numpy as np
from skimage import io
from skimage import segmentation as sg
from skimage import transform as transf
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Zoomout_Vgg16:
    def __init__(self, vgg16_npy_path=None,zlayers=["conv1_1","conv1_2","conv2_1","conv2_2","conv3_1","conv3_2","conv3_3","conv4_1","conv4_2","conv4_3","conv5_1","conv5_2","conv5_3","relu7"],downsample=4,weight=224,height=224):
        self.zlayers = zlayers
        self.zlayers_num = len(self.zlayers)
        self.net = {}
        self.strides={"conv1_1":1,"conv1_2":1,"pool1":2,
                     "conv2_1":2,"conv2_2":2,"pool2":4,
                     "conv3_1":4,"conv3_2":4,"conv3_3":4,"pool3":8,
                     "conv4_1":8,"conv4_2":8,"conv4_3":8,"pool4":16,
                     "conv5_1":16,"conv5_2":16,"conv5_3":16,"pool5":32,
                    }
        self.downsample = downsample
        self.w = weight
        self.h = height
        self.w_d = int(weight / downsample)
        self.h_d = int(height / downsample)
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")
        self.build()
        self.zoomout_features(self.downsample)

    def build(self):
        self.net["input"] = tf.placeholder(shape=[None,self.w,self.h,3],dtype=tf.float32)

        # superpixel mean cal
        self.sp_mean()

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.net["input"])
        assert red.get_shape().as_list()[1:] == [self.w, self.h, 1]
        assert green.get_shape().as_list()[1:] == [self.w, self.h, 1]
        assert blue.get_shape().as_list()[1:] == [self.w, self.h, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [self.w, self.h, 3]

        self.net["conv1_1"] = self.conv_layer(bgr, "conv1_1")
        self.net["conv1_2"] = self.conv_layer(self.net["conv1_1"], "conv1_2")
        self.net["pool1"] = self.max_pool(self.net["conv1_2"], 'pool1')

        self.net["conv2_1"] = self.conv_layer(self.net["pool1"], "conv2_1")
        self.net["conv2_2"] = self.conv_layer(self.net["conv2_1"], "conv2_2")
        self.net["pool2"] = self.max_pool(self.net["conv2_2"], 'pool2')

        self.net["conv3_1"] = self.conv_layer(self.net["pool2"], "conv3_1")
        self.net["conv3_2"] = self.conv_layer(self.net["conv3_1"], "conv3_2")
        self.net["conv3_3"] = self.conv_layer(self.net["conv3_2"], "conv3_3")
        self.net["pool3"] = self.max_pool(self.net["conv3_3"], 'pool3')

        self.net["conv4_1"] = self.conv_layer(self.net["pool3"], "conv4_1")
        self.net["conv4_2"] = self.conv_layer(self.net["conv4_1"], "conv4_2")
        self.net["conv4_3"] = self.conv_layer(self.net["conv4_2"], "conv4_3")
        self.net["pool4"] = self.max_pool(self.net["conv4_3"], 'pool4')

        self.net["conv5_1"] = self.conv_layer(self.net["pool4"], "conv5_1")
        self.net["conv5_2"] = self.conv_layer(self.net["conv5_1"], "conv5_2")
        self.net["conv5_3"] = self.conv_layer(self.net["conv5_2"], "conv5_3")
        self.net["pool5"] = self.max_pool(self.net["conv5_3"], 'pool5')

        self.net["fc6"] = self.fc_layer(self.net["pool5"], "fc6")
        assert self.net["fc6"].get_shape().as_list()[1:] == [4096]
        self.net["relu6"] = tf.nn.relu(self.net["fc6"])

        self.net["fc7"] = self.fc_layer(self.net["relu6"], "fc7")
        self.net["relu7"] = tf.nn.relu(self.net["fc7"])

        self.net["fc8"] = self.fc_layer(self.net["relu7"], "fc8")

        self.net["output"] = tf.nn.softmax(self.net["fc8"], name="prob")


    def sp_mean(self):
        self.net["sp_input"] = tf.placeholder(shape=[None,self.w_d*self.h_d],dtype=tf.float32)
        self.net["sp_mask" ] = tf.placeholder(shape=[self.w_d*self.h_d],dtype=tf.float32)
        self.net["sp_mean"] = tf.reduce_mean(self.net["sp_input"]*self.net["sp_mask"],0)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def zoomout_features(self,downsample = 4):
        self.zoomout_f = {}
        for layer in self.zlayers:
            if layer.startswith("conv"): # the features after conv layer
                if self.strides[layer] < downsample: # the stride of the layer is less than the target stride
                    scale = downsample/self.strides[layer]
                    self.zoomout_f[layer] = tf.nn.max_pool(self.net[layer],ksize=[1,1,1,1],strides=[1,scale,scale,1],padding="SAME")
                else:
                    scale = self.strides[layer] / downsample
                    im_w = int(self.net[layer].get_shape().as_list()[1] * scale)
                    im_h = int(self.net[layer].get_shape().as_list()[2] * scale)
                    self.zoomout_f[layer] = tf.image.resize_images(self.net[layer],[im_w,im_h])
                    print("layer:%s,shape:%s" % (layer,self.zoomout_f[layer].get_shape()))
            else: # the fc layer
                self.zoomout_f[layer] = self.net[layer]
                print("layer:%s,shape:%s" % (layer,self.zoomout_f[layer].get_shape()))

    def get_zoomout_features(self,sess,imgs,superpixel_imgs):
        #print("in1")
        batch_num = len(superpixel_imgs)
        zoomout_f_list = [ self.zoomout_f[layer] for layer in self.zlayers ]
        zoomout_tmp = sess.run(zoomout_f_list, feed_dict = {self.net["input"]:imgs}) # shape = [zlayers_num, batch, w, h] ,note each elements is a k-dims list, and k is not a constant number.
        zoomout_f_tmp = None
        zoomout_s_tmp = None
        #print("in2")
        for i,layer in enumerate(self.zlayers):
            if layer.startswith("conv"):
                if zoomout_f_tmp is not None:
                    zoomout_f_tmp = np.concatenate((zoomout_f_tmp,zoomout_tmp[i]),axis=3)
                else:
                    zoomout_f_tmp = zoomout_tmp[i]
            else:
                if zoomout_s_tmp is not None:
                    zoomout_s_tmp = np.concatenate((zoomout_s_tmp, zoomout_tmp[i]), axis=1)
                else:
                    zoomout_s_tmp = zoomout_tmp[i]
        #print("zoomout_s_tmp shape:%s" % str(zoomout_s_tmp.shape))
        #print("in3")
        #zoomout_f_tmp = np.reshape(zoomout_f_tmp,[batch_num,self.w * self.h, -1])
        #print("zoomout_f_tmp shape:%s" % str(zoomout_f_tmp.shape))

        zoomout_f_count = zoomout_f_tmp.shape[3]
        zoomout_f_ones = np.ones([zoomout_f_count])
        #print("ones:%s" % str(zoomout_f_ones))
        ret_f = []
        for batch_index in range(batch_num):
            #print("batch_index:%d" % batch_index)
            superpixel_img = superpixel_imgs[batch_index]
            superpixel_num = max(superpixel_img.reshape([-1])) + 1
            # get subscene features
            subscene_spatial_info = {} 
            subscene_img_info = {} 
            for w_index in range(self.w_d): # the map after the pool layers
                for h_index in range(self.h_d):
                    sp_index = superpixel_img[w_index][h_index]
                    key = str(sp_index)
                    if key not in subscene_spatial_info:
                        subscene_spatial_info[key] = set([])
                        subscene_img_info[key] = {"img":np.zeros(shape=superpixel_img.shape),"left":w_index,"right":w_index,"top":h_index,"bottom":h_index}
                    # update the neighbor information
                    if h_index+1 < self.h_d:
                        subscene_spatial_info[key].add(superpixel_img[w_index][h_index+1])
                    if h_index-1 >= 0:
                        subscene_spatial_info[key].add(superpixel_img[w_index][h_index-1])
                    if w_index+1 < self.w_d:
                        subscene_spatial_info[key].add(superpixel_img[w_index+1][h_index])
                    if w_index-1 >= 0:
                        subscene_spatial_info[key].add(superpixel_img[w_index-1][h_index])
                    # update the superpixel region infomation
                    subscene_img_info[key]["img"][w_index][h_index] = 1
                    if w_index < subscene_img_info[key]["left"]:
                        subscene_img_info[key]["left"] = w_index
                    if w_index > subscene_img_info[key]["right"]:
                        subscene_img_info[key]["right"] = w_index
                    if h_index < subscene_img_info[key]["top"]:
                        subscene_img_info[key]["top"] = h_index
                    if h_index > subscene_img_info[key]["bottom"]:
                        subscene_img_info[key]["bottom"] = h_index
           #print("sp img:%s" % str(superpixel_img))
           #print("sp img:%s" % str(superpixel_img[0]))
           #print("sp img:%s" % str(superpixel_img[1]))
           #print("sp img:%s" % str(superpixel_img[2]))
           #print("sp img:%s" % str(superpixel_img[3]))
           #print("sp img:%s" % str(superpixel_img[4]))
           #print("sp img:%s" % str(superpixel_img[5]))
           #print("sp img:%s" % str(superpixel_img[6]))
           #print("sp img:%s" % str(superpixel_img[7]))
           #print("sp img:%s" % str(superpixel_img[8]))
           #print("subscene spatial info:%s" % str(subscene_spatial_info))
            subscene_3_spatial_info = {} # get the 3 radius neighbors information
            for one in range(superpixel_num):
                key_1 = str(one)
                subscene_3_spatial_info[key_1] = set([one]) # radius 0
                subscene_3_spatial_info[key_1].update(subscene_spatial_info[key_1]) # radius 1
                for two in subscene_spatial_info[key_1]:
                    key_2 = str(two)
                    subscene_3_spatial_info[key_1].update(subscene_spatial_info[key_2]) # radius 2
                    for three in subscene_spatial_info[key_2]:
                        key_3 = str(three)
                        subscene_3_spatial_info[key_1].update(subscene_spatial_info[key_3]) # radius 3
            subscene_batch = np.zeros(shape=[superpixel_num,4096])
            for one in range(superpixel_num):
                sp_index = str(one)
                left = min([subscene_img_info[str(o)]["left"] for o in subscene_3_spatial_info[sp_index]])
                right = max([subscene_img_info[str(o)]["right"] for o in subscene_3_spatial_info[sp_index]])
                top = min([subscene_img_info[str(o)]["top"] for o in subscene_3_spatial_info[sp_index]])
                bottom = min([subscene_img_info[str(o)]["bottom"] for o in subscene_3_spatial_info[sp_index]])
                mask = None
                for n in subscene_3_spatial_info[sp_index]:
                    if mask is None: mask = subscene_img_info[str(n)]["img"]
                    else: mask += subscene_img_info[str(n)]["img"]
                mask = np.reshape(mask,(112,112,1))
                mask = np.tile(mask,(1,1,3))
                imgs_tmp = transf.resize(imgs[batch_index],[112,112])
                sp_img_with_neighbor_tmp = imgs_tmp * mask
                sp_img_with_neighbor = transf.resize(sp_img_with_neighbor_tmp[top:bottom+1,left:right+1],[224,224])
                subscene_batch[one] = sess.run(self.zoomout_f["relu7"], feed_dict = {self.net["input"]:[sp_img_with_neighbor]})



            # get the zoomout features for convolution layers
            zoomout_batch = np.zeros(shape=[2,superpixel_num,zoomout_f_count])
            for w_index in range(self.w_d): # create the zoomout features
                for h_index in range(self.h_d):
                   #print("sp num:%d" % superpixel_num) 
                   #print("w:%d" % self.w_d) 
                   #print("h:%d" % self.h_d) 
                   #print("shape:%s" % str(superpixel_img.shape))
                   #print("sp index:%d" % superpixel_img[w_index][h_index])
                   #print("zoomout_f shape:%s" % str(zoomout_f_tmp.shape))
                    zoomout_batch[0][superpixel_img[w_index][h_index]] += zoomout_f_ones
                    zoomout_batch[1][superpixel_img[w_index][h_index]] += zoomout_f_tmp[batch_index][w_index][h_index]

            zoomout_batch[1] /= zoomout_batch[0]

            

            zoomout_s_tmp_batch = np.tile(zoomout_s_tmp[batch_index],[superpixel_num,1])
            zoomout_batch = np.concatenate((zoomout_batch[1],subscene_batch,zoomout_s_tmp_batch),axis=1)
            #print("shape:%s" % str(zoomout_batch.shape))
            ret_f.append(zoomout_batch)
        return ret_f

if __name__ == "__main__":
    zoomout = Zoomout_Vgg16("vgg16.npy")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img = io.imread("test.jpg")
    img = transf.resize(img,(224,224))
    img_s = transf.resize(img,(56,56))
    superpixel_img = sg.slic(img_s,100)
    zoomout.get_zoomout_features(sess,[img],[superpixel_img],100)

