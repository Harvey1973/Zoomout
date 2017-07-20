# coding=utf-8

import scipy.misc as misc
import sys

image_name = sys.argv[1]
a = misc.imread(image_name)
#a = misc.imread("2007_000032.png",mode="L")
l = set([])
rate = {}
s = a.shape
print("s:%s" % str(s))
for i in range(s[0]):
    for k in range(s[1]):
        one = a[i][k]
        #if not isinstance(one,int): # the image is rgb
        #if len(one) > 1:
            #print("type:%s" % type(one))
            #one = "%d_%d_%d" % (one[0],one[1],one[2]) 
        if one not in l:
            l.add(one) 
            rate[str(one)] = 0
        else:
            rate[str(one)] += 1
        #a[i][k] = one*40
#for one in a:
   #print(one)
print("len:%d" % len(l))
print(l)
print("rate:%s" % rate)
#misc.imsave("test.png",a)
