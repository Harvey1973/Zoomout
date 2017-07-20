import scipy.io as io
import skimage.io as sio
import numpy as np
import os
import sys

usage = "python xxx.py slic_mat_path output_mat_path"

image_w = 224
image_h = 224
main_path = "output"

slic_mat_path = sys.argv[1]
slic_mat = io.loadmat(slic_mat_path)
slic_data = slic_mat.get("test_slic")
output_mat_path = sys.argv[2]
output_mat = io.loadmat(output_mat_path)
output_data = output_mat

images_num = slic_data.shape[0]
sp_index = 0
for i in range(images_num):
    print("start %dth image..." % i)
    gt_image = np.zeros([image_w,image_h])
    pred_image = np.zeros([image_w,image_h])
    sp_num = max(slic_data[i].reshape([-1])) + 1
    for j in range(sp_num):
        slic_mask = np.ma.masked_equal(slic_data[i],j).mask.astype(int)
        gt_image += slic_mask*output_data["gt"][sp_index]
        pred_image += slic_mask*output_data["pred"][sp_index]
        sp_index += 1
    gt_image = gt_image.astype(int)
    pred_image = pred_image.astype(int)
    print("gt_image:%s" % str(gt_image))
    sio.imsave(os.path.join(main_path,"%d_gt.png" % i),gt_image)
    sio.imsave(os.path.join(main_path,"%d_pred.png" % i),pred_image)
