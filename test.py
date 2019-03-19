from module import *
from PIL import Image
import os


def read_img(rootdir):
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    count=0
    right_count=0
    for i in range(0, len(list)):
        if list[i] != ".DS_Store":
            path = os.path.join(rootdir, list[i])
            text = list[i].split(".")[0]  # 取文件名-验证码结果

            img = Image.open(path)

            result=image_recognition(img)
            if(text==result):
                right_count+=1
            count+=1

            print("正确："+text+",预测："+result)
    print("正确率："+str(right_count*(1.0)/count))

read_img("old")
