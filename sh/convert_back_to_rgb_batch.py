# coding=utf-8

import scipy.misc as misc
import json
import numpy as np
import os
import sys

f = open("/home/zhangzhengqiang/git/bls/data/convert.txt","r")
l = json.loads(f.readline())
f.close()
l = eval(l.strip("\n"))
b = {}
for key in l:
    rgb = key.split("_") 
    b[str(l[key])] = rgb

def get_rgb(gray):
    if str(gray) not in b:
        print("error: new color")
    return b[str(gray)]

for j in range(int(sys.argv[2])):
    print("deal with %s ..." % j)
    tmp = misc.imread(os.path.join(sys.argv[1],"%d_pred.png" % j))
    s = tmp.shape
    #print "tmp:",s
    tmp_ = np.full([s[0],s[1],3],0)
    for i in range(s[0]):
        for k in range(s[1]):
            one = tmp[i][k]
            tmp_[i][k]= get_rgb(one)
            
    misc.imsave(os.path.join(sys.argv[1],"convert","%d_pred_color.png" % j),tmp_)

    tmp = misc.imread(os.path.join(sys.argv[1],"%d_gt.png" % j))
    s = tmp.shape
    #print "tmp:",s
    tmp_ = np.full([s[0],s[1],3],0)
    for i in range(s[0]):
        for k in range(s[1]):
            one = tmp[i][k]
            tmp_[i][k]= get_rgb(one)
            
    misc.imsave(os.path.join(sys.argv[1],"convert","%d_gt_color.png" % j),tmp_)

#f.close()
