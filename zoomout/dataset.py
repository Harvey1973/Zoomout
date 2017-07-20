import skimage.io as io
import skimage.transform as transf
import skimage.segmentation as sg
import skimage
import os
import numpy as np

class data:
    def __init__(self,main_path="data",sp_num=50,weight=224,height=224,downsample=4):
        self.main_path = main_path
        self.sp_num = sp_num
        self.w = weight
        self.h = height
        self.w_ = self.w // downsample
        self.h_ = self.h // downsample
        self.update_list()
        self.create_data()

    def update_list(self):
        self.list = {}
        self.len_of_data = {}
        for kind in ["train","test"]:
            self.list[kind] = self.update_list_(kind)
            self.len_of_data[kind] = len(self.list[kind])

    def update_list_(self,kind):
        # train
        list_ = []
        f = open(os.path.join(self.main_path,"txt","%s.txt" % kind),"r")
        for line in f.readlines():
            line = line.strip("\n")
            list_.append(line)
        f.close()
        return list_

    def create_data(self):
        self.index = {"train":0,"test":0}
        self.data = {}
        for kind in ["train","test"]:
            self.data[kind] = self.create_data_(kind)
        
    def create_data_(self,kind):
        data = {"images":[],"annotations":[],"slic":[]}
        for one_im in self.list[kind]:
            #print("list:%s" % str(self.list[kind]))
            image_f_path = os.path.join(self.main_path,"images","%s.jpg" % one_im)
            image_f = io.imread(image_f_path)
            image_f = transf.resize(image_f,(self.w,self.h))
            data["images"].append(skimage.img_as_ubyte(image_f))
  
            image_f_t = transf.resize(image_f,(self.w_,self.h_))
            image_slic = sg.slic(image_f_t,self.sp_num,15)
            data["slic"].append(image_slic)

            label_f_path = os.path.join(self.main_path,"annotations","%s.png" % one_im)
            label_f = io.imread(label_f_path)
            label_f = transf.resize(label_f,(self.w_,self.h_))
            label_f = skimage.img_as_ubyte(label_f)
            label_features = []
            #tmp = image_slic.astype(float)
            #tmp = tmp / 500
            #tmp = transf.resize(tmp,(224,224))
            #tmp = tmp * 500
            #tmp = tmp.astype(int)
            #io.imsave("slic.png",tmp)
            #tmp = transf.resize(label_f,(224,224))
            #tmp = skimage.img_as_ubyte(tmp)
            #io.imsave("label.png",tmp)
            superpixel_num = max(image_slic.reshape([-1])) + 1
            for i in range(superpixel_num):
                mask = np.ma.masked_equal(image_slic,i).mask.astype(int)
                if len(mask.shape) <= 1:
                    #print("continue")
                    label_features.append(0)
                    continue
                #io.imsave("%d_mask.png" % i,mask)
                #io.imsave("%d_mask_ret.png" % i,mask_ret)
                count = np.bincount(label_f.reshape([-1]), weights=mask.reshape([-1]))
                feature = np.argmax(count)
                #print("feature:%f" % feature)
                label_features.append(feature)
            data["annotations"].append(label_features)
        return data
                
    def next_batch(self,batch,kind="train"):
        print("in next batch")
        index_s = self.index[kind]
        index_e = index_s + batch
        if index_e >= self.len_of_data[kind]:
            ret_data = [self.data[kind]["images"][index_s:],self.data[kind]["annotations"][index_s:],self.data[kind]["slic"][index_s:]]
            overflow = True
        else:
            #print("kind:%s,data keys:%s" % (kind, self.data.keys()))
            #print("train keys:%s" % (self.data["train"].keys()))
            ret_data = [self.data[kind]["images"][index_s:index_e],self.data[kind]["annotations"][index_s:index_e],self.data[kind]["slic"][index_s:index_e]]
            overflow = False
        self.index[kind] = index_e
        print("out next batch")
        return ret_data,overflow
       
