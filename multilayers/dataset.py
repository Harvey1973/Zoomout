import scipy.io as io 
from datetime import datetime
import pickle
import sklearn.preprocessing
Scaler = sklearn.preprocessing.StandardScaler
import numpy as np

class dataset():
    #def __init__(self,mat_files={"train":["data/zoomout_1001_1200.mat","data/zoomout_1801_2000.mat","data/zoomout_2201_2400.mat","data/zoomout_601_800.mat","data/zoomout_1_200.mat","data/zoomout_1401_1600.mat","data/zoomout_2001_2200.mat","data/zoomout_2401_2500.mat","data/zoomout_801_1000.mat","data/zoomout_1201_1400.mat","data/zoomout_1601_1800.mat","data/zoomout_201_400.mat","data/zoomout_401_600.mat"],"test":["data/zoomout_test.mat"]}):
    def __init__(self,mat_files={"train":["data4/train_zoomout_0.mat","data4/train_zoomout_1.mat","data4/train_zoomout_2.mat","data4/train_zoomout_3.mat","data4/train_zoomout_4.mat","data4/train_zoomout_5.mat","data4/train_zoomout_6.mat","data4/train_zoomout_7.mat","data4/train_zoomout_8.mat","data4/train_zoomout_9.mat","data4/train_zoomout_10.mat","data4/train_zoomout_11.mat","data4/train_zoomout_12.mat","data4/train_zoomout_13.mat","data4/train_zoomout_14.mat","data4/train_zoomout_15.mat","data4/train_zoomout_16.mat","data4/train_zoomout_17.mat","data4/train_zoomout_18.mat","data4/train_zoomout_19.mat","data4/train_zoomout_20.mat","data4/train_zoomout_21.mat","data4/train_zoomout_22.mat","data4/train_zoomout_23.mat"],"test":["data4/test_zoomout_0.mat","data4/test_zoomout_1.mat","data4/test_zoomout_2.mat"]},scaler=None):
    #def __init__(self,mat_files={"train":["data4/train_zoomout_0.mat","data4/train_zoomout_1.mat"],"test":["data4/test_zoomout_0.mat"]}):
        self.data = {"train":{"x":None,"y":None},"test":{"x":None,"y":None}}
        self.length = {"train":0,"test":0}
        self.mat_files = mat_files
        self.get_data_from_mat()
        self.index = {"train":0,"test":0}
        self.epochs = {"train":0,"test":0}
        self.scaler = scaler
        self.data_preproce()
        self.data_perm = {}
        for category in ["train","test"]:
            perm = np.arange(self.length[category])
            #np.random.shuffle(perm)
            self.data_perm[category] = perm

    def get_data_from_mat(self):
        for category in ["train","test"]:
            for i,mat_file in enumerate(self.mat_files[category]):
                print("mat file:%s" % mat_file)
                if ("%s_zoomout" % category) in mat_file: # to deal with different mat format
                    k_s = "%s_%s_%d" % (category,"%s",i)
                else:
                    k_s = "%s_%s" % (category,"%s")
                mat_data = io.loadmat(mat_file)
                for key in ["x","y"]:
                    if self.data[category][key] is None:
                        self.data[category][key] = mat_data[k_s % key]
                    else:
                        self.data[category][key] = np.concatenate([self.data[category][key],mat_data[k_s % key]],axis=0)
                self.length[category] += mat_data[k_s % "y"].shape[0]


    def get_cur_epoch(self,category="train"):
        return self.epochs[category]

    def next_batch(self,batch_num = 10,category="train"):
        start_index = self.index[category]
        end_index = start_index + batch_num
        if end_index >= self.length[category]:
            start_index = 0
            end_index = batch_num
            perm = np.arange(self.length[category])
            np.random.shuffle(perm)
            self.data_perm[category] = perm
            self.epochs[category] += 1
        self.index[category] = end_index
        perm = self.data_perm[category][start_index:end_index]
        #print("perm:%s" % perm)
        #print("len:%s" % self.length)
        #print("len x:%s" % len(self.data[category]["x"]))
        #print("len y:%s" % len(self.data[category]["y"]))
        return self.data[category]["x"][perm],self.data[category]["y"][perm]

    def get_histogram(self,category="train"):
        histogram = {}
        for i in range(self.length[category]):
            label = np.argmax(self.data[category]["y"][i])
            key = str(label)
            if key not in histogram:
                histogram[key] = 1
            else:
                histogram[key] += 1
        return histogram
    
    def get_length(self,category="train"):
        return self.length[category]

    def decimate_bg_sp(self,rate,category="train"):
        length = self.length[category]
        perm = list(range(length))
        del_count = 0
        for i in range(length):
            if np.argmax(self.data[category]["y"][i]) == 0:
                if np.random.random() < rate: 
                    del perm[i-del_count]
                    del_count += 1
        self.data[category]["x"] = self.data[category]["x"][perm]
        self.data[category]["y"] = self.data[category]["y"][perm]
        self.length[category] = len(perm)

    def data_preproce(self,category="train"):
        if self.scaler is None: 
            self.scaler = Scaler()
            print("before %s" % str(self.data["train"]["x"][0]))
            print("max:%f, min:%f" % (max(self.data["train"]["x"][0]),min(self.data["train"]["x"][0])))
            self.data["train"]["x"] = self.scaler.fit_transform(self.data["train"]["x"])
            print("after  %s" % str(self.data["train"]["x"][0]))
            print("max:%f, min:%f" % (max(self.data["train"]["x"][0]),min(self.data["train"]["x"][0])))
        else:
            print("before %s" % str(self.data["train"]["x"][0]))
            print("max:%f, min:%f" % (max(self.data["train"]["x"][0]),min(self.data["train"]["x"][0])))
            self.data["train"]["x"] = self.scaler.transform(self.data["train"]["x"])
            print("after  %s" % str(self.data["train"]["x"][0]))
            print("max:%f, min:%f" % (max(self.data["train"]["x"][0]),min(self.data["train"]["x"][0])))

        print("before %s" % str(self.data["test"]["x"][0]))
        print("max:%f, min:%f" % (max(self.data["test"]["x"][0]),min(self.data["test"]["x"][0])))
        self.data["test"]["x"] = self.scaler.transform(self.data["test"]["x"])
        print("after  %s" % str(self.data["test"]["x"][0]))
        print("max:%f, min:%f" % (max(self.data["test"]["x"][0]),min(self.data["test"]["x"][0])))

    def save_scaler(self,filename=None):
        if filename is None: filename = "saver/data_scaler_%s" % datetime.now().strftime("%m-%d-%H-%M")
        f = open(filename,"wb")
        pickle.dump(self.scaler,f)
        f.close()


if __name__ == "__main__":
    #d = dataset(mat_files={"train":["data/zoomout_1001_1200.mat","data/zoomout_1801_2000.mat","data/zoomout_2201_2400.mat","data/zoomout_601_800.mat"],"test":["data/zoomout_test.mat"]})
    d=dataset()
    print("length:%s" % str(d.get_length()))
    histogram = d.get_histogram()
    for key in histogram:
        print("key:%s,count:%s" % (key,histogram[key]))
    print("\n\nafter decimation ... \n\n")
    histogram = d.get_histogram()
    for key in histogram:
        print("key:%s,count:%s" % (key,histogram[key]))
    d.save_scaler()
    print("save sacler")
