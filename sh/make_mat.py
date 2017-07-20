import scipy.io
from PIL import Image
import numpy as np
import os

main_path = "data/"
files_path = {"train":os.path.join(main_path,"txt","train.txt"),"test":os.path.join(main_path,"txt","test.txt")}
data = {}
num_of_classes = 21
tmp_mat_file_name = []
for one in files_path:
    key_x = "%s_x" % one
    key_y = "%s_y" % one
    print("file:%s" % files_path[one])
    i = 0
    f = open(files_path[one])
    for line in f.readlines():
        if i % 10 == 0: # store the tmp.mat every 10 iterations
            data = {}
            data[key_x] = []
            data[key_y] = []
        line = line.strip("\n")
        print("delwith %s" % line)
        x_image_path = os.path.join(main_path,"images","%s.jpg" % line)
        x_image = Image.open(x_image_path)
        x_image = x_image.resize((224,224))
        y_image_path = os.path.join(main_path,"annotations","%s.png" % line)
        y_image = Image.open(y_image_path)
        y_image = y_image.resize((224,224))
        y_data = y_image.load()
        crop_w = 32
        crop_h = 32
        for left in range(0,224-crop_w,20):
            #print("left:%d" % left)
            for upper in range(0,224-crop_h,20):
                x_data = x_image.crop([left,upper,left+crop_w,upper+crop_h])
                x_data = np.array(x_data)
                x_data = np.reshape(x_data,[-1,3])
                x_data_tmp = []
                x_data_tmp.extend([on[0] for on in x_data])
                x_data_tmp.extend([on[1] for on in x_data])
                x_data_tmp.extend([on[2] for on in x_data])
                x_data_tmp = [x_data_tmp]
                #print("x_data_tmp:%s" % x_data_tmp)
                if len(data[key_x]) <= 0:data[key_x] = x_data_tmp
                else:data[key_x] = np.concatenate((data[key_x],x_data_tmp))
                
                y_data_tmp = np.zeros(num_of_classes)
                #print("y:%d" % y_data[left+crop_w/2,upper+crop_h/2])
                y_data_tmp[y_data[left+crop_w/2,upper+crop_h/2]] = 1
                y_data_tmp = [y_data_tmp]
                if len(data[key_y]) <= 0:data[key_y] = y_data_tmp
                else:data[key_y] = np.concatenate((data[key_y],y_data_tmp))
        if i % 10 == 9: # store the tmp.mat
            tmp_mat_file = "tmp_%d.mat" % i
            scipy.io.savemat(tmp_mat_file,data)
            tmp_mat_file_name.append(tmp_mat_file)
            print("save the %s." % tmp_mat_file)
        i += 1
    print("start the fusion")
    tmp_mat_data = {}
    for one_tmp in tmp_mat_file_name:
        tmp_mat = scipy.io.loadmat(one_tmp)
        for key in [key_x,key_y]:
            if key not in tmp_mat_data: tmp_mat_data[key] = tmp_mat[key]
            else: tmp_mat_data[key] = np.concatenate((tmp_mat_data[key],tmp_mat[key]))
        os.system("rm %s" % one_tmp)
    
    output_mat_file = "pascal_voc_%s.mat" % one
    scipy.io.savemat(output_mat_file,tmp_mat_data)
    tmp_mat_file_name = []
    f.close()

print("start the final fusion")
tmp_mat_data = {}
for one_tmp in ["pascal_voc_train.mat","pascal_voc_test.mat"]:
    tmp_mat = scipy.io.loadmat(one_tmp)
    for key in ["train_x","train_y","test_x","test_y"]:
        if key in tmp_mat: tmp_mat_data[key] = tmp_mat[key]
    os.system("rm %s" % one_tmp)
output_mat_file = "pascal_voc.mat"
scipy.io.savemat(output_mat_file,tmp_mat_data)
    
        
