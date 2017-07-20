import scipy.io
import scipy.misc as misc
import numpy as np
import os

main_path = "data/"
files_path = {"train":os.path.join(main_path,"txt","train.txt"),"test":os.path.join(main_path,"txt","test.txt")}
data = {}
for one in files_path:
    key_x = "%s_x" % one
    data[key_x] = []
    key_y = "%s_y" % one
    data[key_y] = []
    print("file:%s" % files_path[one])
    f = open(files_path[one])
    for line in f.readlines():
        line = line.strip("\n")
        print("delwith %s" % line)
        x_image_path = os.path.join(main_path,"images","%s.jpg" % line)
        x_image = misc.imread(x_image_path)
        x_image = misc.imresize(x_image,(32,32))
        x_data = x_image.reshape([-1,3]) 
        x_data_tmp = []
        x_data_tmp.extend([x[0] for x in x_data])
        x_data_tmp.extend([x[1] for x in x_data])
        x_data_tmp.extend([x[2] for x in x_data])
        x_data_tmp = [x_data_tmp]
        if len(data[key_x]) <= 0:data[key_x] = x_data_tmp
        else:data[key_x] = np.concatenate((data[key_x],x_data_tmp))

        y_image_path = os.path.join(main_path,"annotations","%s.png" % line)
        y_image = misc.imread(y_image_path)
        y_image = misc.imresize(y_image,(32,32))
        y_data = [y_image.reshape([-1])]
        if len(data[key_y]) <= 0:data[key_y] = y_data
        else:data[key_y] = np.concatenate((data[key_y],y_data))
    f.close()
        
output_mat_file = "pascal_voc_without_one_hot.mat"
scipy.io.savemat(output_mat_file,data)
