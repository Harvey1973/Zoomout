import scipy.io as io
import skimage.io as sio
import numpy as np
import os
import sys

usage = "python xxx.py slic_mat_path output_mat_path"

downsample = 2
image_w = int(224 / downsample)
image_h = int(224 / downsample)
main_path = "output"

slic_mat_path = sys.argv[1]
slic_mat = io.loadmat(slic_mat_path)
slic_data = slic_mat.get("test_slic")
output_mat_path = sys.argv[2]
output_mat = io.loadmat(output_mat_path)
output_data = {"gt":output_mat["gt"][0],"pred":output_mat["pred"][0]}

images_num = slic_data.shape[0]
pred_index = 0
pred_len = len(output_data["gt"])
for i in range(images_num):
    print("start %dth image..." % i)
    gt_image = np.zeros([image_w,image_h])
    pred_image = np.zeros([image_w,image_h])
    sp_num = max(slic_data[i].reshape([-1])) + 1
    for sp_index in range(sp_num):
        slic_mask = np.ma.masked_equal(slic_data[i],sp_index).mask.astype(int)
        gt_image += slic_mask*output_data["gt"][pred_index]
        pred_image += slic_mask*output_data["pred"][pred_index]
        pred_index += 1
        if pred_index >= pred_len:
            break
    if pred_index >= pred_len:
        break
    gt_image = gt_image.astype(int)
    pred_image = pred_image.astype(int)
    print("gt_image:%s" % str(gt_image))
    sio.imsave(os.path.join(main_path,"%d_gt.png" % i),gt_image)
    sio.imsave(os.path.join(main_path,"%d_pred.png" % i),pred_image)
