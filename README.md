# image_recognith
+ 一个使用CNN进行验证码识别的程序（基于TensorFlow）
+ 这个神经网络不进行验证码的分割，直接进行训练，原始图像经过灰度处理
+ train中为训练集，大约3.5w张图片
+ 其中包含一个进行了7200轮训练的模型，测试集准确率87%左右
+ label_image是一个验证标注的小工具，限制输入字母以及文字长度，回车自动重命名以及移动文件夹

================================================================================

+ 尝试在图片灰度处理之后，进行二值化处理，肉眼观察效果还可以，但是训练一直不收敛
