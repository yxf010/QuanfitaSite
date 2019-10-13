---
title: Windows10+Anaconda+TensorFlow(CPU and GPU)环境快速搭建
categories: 
- DeepLearning
tags: 
- TensorFlow
- 环境搭建
copyright: true
---

今天分享一下本人在笔记本上配置TensorFlow环境的过程。

## 说明

**电脑配置：**

- Acer笔记本
- CPU Inter Core i5-6200U
- GPU NVIDIA GeForce 940M(忽略掉我的渣渣GPU)
- Windows10

**所需的环境**：

- Anaconda3(64bit)
- CUDA-8.0
- CuDNN-5.1
- Python-3.6
- TensorFlow 或者 TensorFlow-gpu

## 首先安装Anaconda3

​	我们从官网下载(https://www.anaconda.com/download/#windows)，也可以使用我上传百度网盘的版本，链接：https://pan.baidu.com/s/1dGEC57z 密码：2om4
使用Linux的小伙伴可以同样下载Linux版本的Anaconda，之后我会再做补充的。
![这里写图片描述](http://img.blog.csdn.net/20180115192958597?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	下载好后，我们进入安装界面：
![这里写图片描述](http://img.blog.csdn.net/20180115193411481?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180115193429214?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180115193446894?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180115193527498?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	这里，我们把两个都选上，第一个是加入环境变量，因为我之前安装过一次所以这里提示不要重复添加，第二个是默认的Python3.6，让后Install。

![这里写图片描述](http://img.blog.csdn.net/20180115193504823?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	在完成Anaconda的安装后，我们打开Anaconda的命令行(最好用**管理员身份运行**，否则可能会有权限的问题)：

![这里写图片描述](http://img.blog.csdn.net/20180115193543200?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	我们可以看到一个和Windows命令行很像的一个窗口：

![这里写图片描述](http://img.blog.csdn.net/20180115193553329?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 安装CUDA和CuDNN

​	这里为安装GPU版本的TensorFlow做准备，CPU版本可跳过此部分。

​	CUDA是NVIDIA推出的运算平台，CuDNN是专门针对Deep Learning框架设计的一套GPU计算加速方案。虽然在之后用conda命令安装tensorflow-gpu时会自动安装cudatoolkit和cudnn，但是我总觉得自己安装一遍比较放心。

​	我所用的CUDA和CuDNN分享到百度网盘了，链接：https://pan.baidu.com/s/1dGEC57z 密码：2om4

​	先安装CUDA

​	打开首先先解压：

![这里写图片描述](http://img.blog.csdn.net/20180115193612982?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180115193621641?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180115193629591?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180115193637985?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180115193645646?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	这里我们选择自定义，因为我们只安装CUDA

![这里写图片描述](http://img.blog.csdn.net/20180115193653266?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	只选择CUDA其他组件不安装，否则会安装失败

![这里写图片描述](http://img.blog.csdn.net/20180115193705468?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180115193713876?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	这里可能会提示你安装Visual Studio，忽略掉就好了

![这里写图片描述](http://img.blog.csdn.net/20180115193723866?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	然后就开始安装了，等待安装结束就好了。

![这里写图片描述](http://img.blog.csdn.net/20180115193734175?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	解压cudnn的压缩包里面有三个文件夹

![这里写图片描述](http://img.blog.csdn.net/20180115193758216?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	把这三个文件夹复制到你cuda的安装目录下，我的地址是C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0

![这里写图片描述](http://img.blog.csdn.net/20180115193809788?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	这样CUDA和CuDNN就安装好了。

## 创建TensorFlow环境

​	我们在刚刚打开的命令行里输入命令(conda的命令大家可以在这篇博客中找到http://blog.csdn.net/fyuanfena/article/details/52080270)：

> conda create -n tensorflow_gpu python=3.6

![这里写图片描述](http://img.blog.csdn.net/20180115193819785?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	中间会让我们确认一下，输入个y回车就好了。安装好后会给我们提示用activate，和deactivate进行环境的切换。

​	我们先切换到创建好的环境中：

> activate tensorflow_gpu

​	现在，基本环境已经配置好了，我们要安装一些重要的Python科学运算库，Anaconda已经为我们准备好的一系列常用的Python苦，例如numpy，pandas，matplotlib等等，所以我们只需要安装一次anaconda库就可以把这些库全部安装好。

> conda install anaconda

![这里写图片描述](http://img.blog.csdn.net/20180115193832913?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	可以看到，真的有好多常用库。

## 安装TensorFlow

​	之后就是我们最重要的一步，安装TensorFlow：

CPU版本

> conda install tensorflow
> ![这里写图片描述](http://img.blog.csdn.net/20180115193842777?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

GPU版本

> conda install tensorflow-gpu

![这里写图片描述](http://img.blog.csdn.net/20180115193851653?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	这样我们的TensorFlow环境已经配置好了。

## 测试

​	最后，我们进入jupyter notebook(Anaconda自带的Python IDE，自我感觉挺好用的)输入一段官方文档录入的代码测试一下：

​	直接输入jupyter notebook，回车

```python
import tensorflow as tf
hello = tf.constant('Hello,TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

![这里写图片描述](http://img.blog.csdn.net/20180115193901780?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	恭喜，你的TensorFlow已经可以用了，接下来快搭建你自己的神经网络吧~！

## 参考文章

1. [Anaconda常用命令大全](http://blog.csdn.net/fyuanfena/article/details/52080270)
2. [NVIDIA CuDNN 安装说明](https://www.cnblogs.com/platero/p/4118139.html)


