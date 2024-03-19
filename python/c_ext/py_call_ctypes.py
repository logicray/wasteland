from ctypes import *
from ctypes import *    #pip ctypes库，并导入库
test = CDLL("./csrc/sum")    #调用当前目录下叫test的dll文件，dll文件是C生成的动态链接库

print("加载成功")

result =test.sum(5, 10)    #调用库里的函数sum，求和函数
print("result: ",result)    #打印结果
