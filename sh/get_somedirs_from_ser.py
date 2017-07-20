#!/bin/bash

sftp zhangzhengqiang@115.156.245.251 <<CMD
get /home/zhangzhengqiang/git/bls/$1/* /home/xtudbxk/code/bls/$1/ -r 
exit
CMD
