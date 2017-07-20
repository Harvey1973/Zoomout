# coding=utf-8

import scipy.misc as misc
import json
import numpy as np
import os
import sys

f = open("/home/xtudbxk/git/bls/data/convert.txt","r")
l = f.readline()
f.close()
l = eval(l.strip("\n"))
b = {}
for key in l:
    rgb = key.split("_") 
    b[str(l[key])] = rgb

def get_rgb(gray):
    gray = gray % 256
    if str(gray) not in b:
        print("error: new color")
    return b[str(gray)]

tmp = misc.imread(sys.argv[1])
s = tmp.shape
tmp_ = np.full([s[0],s[1],3],0)
for i in range(s[0]):
    for k in range(s[1]):
        one = tmp[i][k]
        tmp_[i][k]= get_rgb(one)
        
path = os.path.splitext(sys.argv[1])
print("path:%s" % str(path))
misc.imsave("%s_rgb.png" % path[0],tmp_)
