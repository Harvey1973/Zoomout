import scipy.io as io
import scipy.misc as misc


sp_num = 50

output_mat = io.loadmat("output.mat")
gt = output_mat["gt"]
pred = output_mat["pred"]

slic_mat = io.loadmat("zoomout.mat")
test_slic = slic_mat["slic"]

count = 0
img_gt = np.zeros(shape=(56,56))
img_pred = np.zeros(shape=(56,56))
for i,one in enumerate(gt):
    if i // sp_num != count: # a new img
        count += 1
        misc.imsave("%_gt.png" % count,img_gt)
        misc.imsave("%_pred.png" % count,img_pred)
        img_gt = np.zeros(shape=(56,56))
        img_pred = np.zeros(shape=(56,56))
        
    num = i % sp_num # iindicate the count of superpixel
    mask = np.ma.masked_equal(test_slic[count],num).mask.astype(int)
    mask_gt = mask * gt[i]
    img_gt = img_gt + mask_gt
    mask_pred = mask * pred[i]
    img_pred = img_pred + mask_pred
