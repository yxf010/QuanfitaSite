今天分享一下本人在笔记本上配置TensorFlow环境的过程。

##说明

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
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkyOTU4NTk3)

​	下载好后，我们进入安装界面：
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNDExNDgx)

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNDI5MjE0)

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNDQ2ODk0)

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNTI3NDk4)

​	这里，我们把两个都选上，第一个是加入环境变量，因为我之前安装过一次所以这里提示不要重复添加，第二个是默认的Python3.6，让后Install。

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNTA0ODIz)

​	在完成Anaconda的安装后，我们打开Anaconda的命令行(最好用**管理员身份运行**，否则可能会有权限的问题)：

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNTQzMjAw)

​	我们可以看到一个和Windows命令行很像的一个窗口：

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNTUzMzI5)

## 安装CUDA和CuDNN

​	这里为安装GPU版本的TensorFlow做准备，CPU版本可跳过此部分。

​	CUDA是NVIDIA推出的运算平台，CuDNN是专门针对Deep Learning框架设计的一套GPU计算加速方案。虽然在之后用conda命令安装tensorflow-gpu时会自动安装cudatoolkit和cudnn，但是我总觉得自己安装一遍比较放心。

​	我所用的CUDA和CuDNN分享到百度网盘了，链接：https://pan.baidu.com/s/1dGEC57z 密码：2om4

​	先安装CUDA

​	打开首先先解压：

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNjEyOTgy)

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNjIxNjQx)

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNjI5NTkx)

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNjM3OTg1)

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNjQ1NjQ2)

​	这里我们选择自定义，因为我们只安装CUDA

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNjUzMjY2)

​	只选择CUDA其他组件不安装，否则会安装失败

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNzA1NDY4)

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNzEzODc2)

​	这里可能会提示你安装Visual Studio，忽略掉就好了

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNzIzODY2)

​	然后就开始安装了，等待安装结束就好了。

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNzM0MTc1)

​	解压cudnn的压缩包里面有三个文件夹

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzNzU4MjE2)

​	把这三个文件夹复制到你cuda的安装目录下，我的地址是C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzODA5Nzg4)

​	这样CUDA和CuDNN就安装好了。

## 创建TensorFlow环境

​	我们在刚刚打开的命令行里输入命令(conda的命令大家可以在这篇博客中找到http://blog.csdn.net/fyuanfena/article/details/52080270)：

> conda create -n tensorflow_gpu python=3.6

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzODE5Nzg1)

​	中间会让我们确认一下，输入个y回车就好了。安装好后会给我们提示用activate，和deactivate进行环境的切换。

​	我们先切换到创建好的环境中：

> activate tensorflow_gpu

​	现在，基本环境已经配置好了，我们要安装一些重要的Python科学运算库，Anaconda已经为我们准备好的一系列常用的Python苦，例如numpy，pandas，matplotlib等等，所以我们只需要安装一次anaconda库就可以把这些库全部安装好。

> conda install anaconda

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzODMyOTEz)

​	可以看到，真的有好多常用库。

##安装TensorFlow

​	之后就是我们最重要的一步，安装TensorFlow：

CPU版本

> conda install tensorflow
![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzODQyNzc3)

GPU版本

> conda install tensorflow-gpu

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzODUxNjUz)

​	这样我们的TensorFlow环境已经配置好了。

##测试

​	最后，我们进入jupyter notebook(Anaconda自带的Python IDE，自我感觉挺好用的)输入一段官方文档录入的代码测试一下：

​	直接输入jupyter notebook，回车

```python
import tensorflow as tf
hello = tf.constant('Hello,TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMTE1MTkzOTAxNzgw)

​	恭喜，你的TensorFlow已经可以用了，接下来快搭建你自己的神经网络吧~！
个人博客：[Quanfita的博客](http://quanfita.cn)


## 参考文章

1. [Anaconda常用命令大全](http://blog.csdn.net/fyuanfena/article/details/52080270)
2. [NVIDIA CuDNN 安装说明](https://www.cnblogs.com/platero/p/4118139.html)




