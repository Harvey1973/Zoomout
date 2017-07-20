import sys
import os

test_f = open("/home/zhangzhengqiang/git/bls/data4/txt/test.txt")
lines = test_f.readlines()
test_f.close()
for i,line in enumerate(lines):
    line = line.rstrip("\n")
    os.system("cp /home/zhangzhengqiang/git/bls/data4/images/%s.jpg /home/zhangzhengqiang/git/bls/experiment/multilayers/inference_result/output/%d.jpg" % (line,i))
