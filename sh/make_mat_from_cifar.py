import scipy.io
import pickle
import os
import numpy as np

data_infos = {"file_prefix_name":"data_batch_","count":5,"main_path":"data"}
output_data = {"train_x":None,"train_y":None,"test_x":None,"test_y":None}
for i in range(data_infos["count"]):
    if i == data_infos["count"] -1:
        x = "train_x"
        y = "train_y"
    else:
        x = "test_x"
        y = "test_y"
    data_file_name = "%s%d" % (data_infos["file_prefix_name"],i+1)
    data_dir = os.path.join(data_infos["main_path"],data_file_name)
    f = open(data_dir,"rb")
    data = pickle.load(f,encoding="bytes")
    f.close()
    if output_data[x] is None: output_data[x]=data[b'data']
    else: output_data[x] = np.concatenate([output_data[x],data[b'data']])
    y_tmp = np.zeros([len(data[b'labels']),10])
    for k,one in enumerate(data[b'labels']):
        y_tmp[k][one] = 1
    if output_data[y] is None: output_data[y]=y_tmp
    else: output_data[y] = np.concatenate([output_data[y],y_tmp])

scipy.io.savemat("cifar.mat",output_data)
