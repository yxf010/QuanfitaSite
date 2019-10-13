---
title: 机器学习入门之《统计学习方法》笔记整理——感知机
categories: 
- note
tags: 
- Machine Learning
- Perceptron
copyright: true
mathjax: true
---
&emsp;&emsp;从头开始学习李航老师的《统计学习方法》，这本书写的很好，非常适合机器学习入门。

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

##感知机模型

&emsp;&emsp;什么是感知机？感知机是二类分类的线性分类模型，其输入为实例的特征向量，输出为实例的类别，取+1和-1二值。感知机学习旨在求出可以将数据进行划分的分离超平面，所以感知机能够解决的问题首先要求特征空间是线性可分的，再者是二类分类，即将样本分为{+1, -1}两类。分离超平面方程为：

$$w·x+b=0$$

&emsp;&emsp;这样，我们就可以构建一个由输入空间到输出空间的函数：

$$f(x)=sign(w·x+b)$$

&emsp;&emsp;称为感知机。其中，w和b为感知机模型的参数，$$w \in R^n$$ 叫作权值（weight），$$b \in R$$ 叫作偏置,$$sign$$ 是符号函数，即

$$sign(x)=\begin{cases} +1 & \text{ , } x\geq 0 \\ -1 & \text{ , } x< 0 \end{cases} $$ 

&emsp;&emsp;感知机模型的假设空间是定义在特征空间中的线性分类模型，即函数集合$$\left \{ f \mid  f(x)=w\cdot x+b \right \}$$ 。

![这里写图片描述](http://img.blog.csdn.net/2018021122084386?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##感知机学习策略

&emsp;&emsp;给定一个数据集$$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$$ ，其中，$$x_i \in X= \mathbb{R}^n$$ ，$$y_i \in Y = \left \{ +1,-1 \right \}$$ ，$$i = 1,2,...,N$$ 。我们假定数据集中所有$$y_i=+1$$ 的实例$$i$$，有$$w\cdot x +b>0$$ ，对所有$$y_i=-1$$ 的实例$$i$$，有$$$w\cdot x +b<0$$ 。

&emsp;&emsp;先给出输入空间$$\mathbb{R}^n$$ 中任意一点$$x_0$$ 到超平面$$S$$ 的距离：

$$-\frac{1}{\left \| w \right \|}\left | (w\cdot x_0+b) \right |$$ 

&emsp;&emsp;这里，$$\left \| w \right \|$$ 是$$w$$ 的$$L_2$$ 范数。

&emsp;&emsp;对于误分类数据$$(x_i,y_i)$$ 来说，有

$$-y_i(w\cdot x_i+b)>0$$ 

&emsp;&emsp;因此，误分类点$$x_i$$ 到超平面$$S$$ 的距离可以写作：

$$-\frac{1}{\left \| w \right \|} y_i (w\cdot x_i+b) $$ 

&emsp;&emsp;假设超平面$$S$$ 的误分类点的集合为$$M$$ ,那么所有误分类点到超平面$$S$$ 的总距离为

$$-\frac{1}{\left \| w \right \|} \sum \limits_{x_i\in M} y_i (w\cdot x_i+b) $$ 

&emsp;&emsp;这里$$\left \| w \right \|$$ 的值是固定的，不必考虑，于是我们就可以得到感知机$$sign(w\cdot x+b)$$ 的损失函数为：

$$L(w,b)=-\sum \limits_{x_i\in M} y_i (w\cdot x_i+b) $$ 

&emsp;&emsp;这个损失函数就是感知机学习的经验风险函数。

##感知机学习算法

&emsp;&emsp;通过上面的损失函数，我们很容易得到目标函数

$$\min \limits_{w,b}L(w,b)=-\sum \limits_{x_i\in M} y_i (w\cdot x_i+b) $$ 

&emsp;&emsp;感知机学习算法是误分类驱动的，具体采用随机梯度下降法( stochastic gradient descent )。

### 原始形式

&emsp;&emsp;所谓原始形式，就是我们用梯度下降的方法，对参数$$w$$ 和$$b$$ 进行不断的迭代更新。任意选取一个超平面$$w_0,b_0$$ 然后使用梯度下降法不断地极小化目标函数。**随机梯度下降的效率要高于批量梯度下降** ( 参考Andrew Ng的[CS229讲义](http://cs229.stanford.edu/notes/cs229-notes1.pdf), Part 1 LMS algorithm部分,[翻译](https://millearninggroup.github.io/Stanford-CS229-CN/translation/cs229-notes1-cn/)) 。

&emsp;&emsp;假设误分类点集合$$M$$ 是固定的，那么损失函数$$L(w,b)$$ 的梯度为

$$\nabla_w L(w,b) = -\sum \limits_{x_i \in M} y_ix_i$$ 

$$\nabla_b L(w,b) = -\sum \limits_{x_i \in M} y_i$$ 

&emsp;&emsp;接下来，随机选取一个误分类点$$(x_i,y_i)$$ ，对$$w,b$$ 进行更新：

$$w \leftarrow w+\eta y_ix_i$$ 

$$b \leftarrow b+\eta y_i$$ 

&emsp;&emsp;其中$$\eta(0<\eta\le1)$$ 为步长，也称学习率（learning rate）。步长越长，下降越快，如果步长过长，会跨过极小点导致发散；如果步长过小，消耗时间会很长。通过迭代，我们的损失函数就不断减小,直到为0。

####算法1  (感知机学习算法的原始形式)

输入: 训练数据集$$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_n,y_n) \right \}$$ , 其中$$x_i \in X = \mathbb{R}^n,y_i \in Y = \left \{ -1,+1 \right \},i = 1,2,...,N$$ ；学习率$$\eta(0<\eta\leq1)$$ ；

输出: $$w,b$$ ；感知机模型$$f(x)=sign(w\cdot x+b)$$ 

(1) 选取初值$$w_0,b_0$$ 

(2) 在训练集中选取数据$$(x_i,y_i)$$ 

(3) 如果$$y_i(w\cdot x+b) \leq 0$$ 

$$w \leftarrow w+\eta y_ix_i$$ 

$$b \leftarrow b+\eta y_i$$ 

(4) 转至 (2) , 直到训练集中没有错误分类点。

&emsp;&emsp;这种学习算法直观上有如下解释：当一个样本被误分类时，就调整w和b的值，使超平面S向误分类点的一侧移动，以减少该误分类点到超平面的距离，直至超平面越过该点使之被正确分类。

**例子：**如图所示，正实例点是$$x_1=(3,3)^T , x_2=(4,3)^T$$ ，负实例点是$$x_3=(1 , 1)^T$$ ，使用感知机算法求解感知机模型$$f(x)=sign(w⋅x+b)$$ 。这里，$$w = (w^{(1)},w^{(2)}),x=(x^{(1)},x^{(2)})$$ 。

![这里写图片描述](http://img.blog.csdn.net/20180211220931175?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

&emsp;&emsp;这里我们取初值$$w_0=0,b_0=0$$ , 取$$\eta=1$$ 。

Python代码如下:

```python
train = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]

w = [0, 0]
b = 0

# 使用梯度下降法更新权重
def update(data):
    global w, b
    w[0] = w[0] + 1 * data[1] * data[0][0]
    w[1] = w[1] + 1 * data[1] * data[0][1]
    b = b + 1 * data[1]
    print(w, b)

# 计算到超平面的距离
def cal(data):
    global w, b
    res = 0
    for i in range(len(data[0])):
        res += data[0][i] * w[i]
    res += b
    res *= data[1]
    return res
 
# 检查是否可以正确分类
def check():
    flag = False
    for data in train:
        if cal(data) <= 0:
            flag = True
            update(data)
    if not flag:
        print("The result: w: " + str(w) + ", b: "+ str(b))
        return False
    flag = False

for i in range(1000):
    check()
    if check() == False:
        break
```

&emsp;&emsp;可以得到如下结果：

> ```
> [3, 3] 1
> [2, 2] 0
> [1, 1] -1
> [0, 0] -2
> [3, 3] -1
> [2, 2] -2
> [1, 1] -3
> The result: w: [1, 1], b: -3
> ```

&emsp;&emsp;如果选取的初值不同或选取不同的误分类点，我们得到的超平面也不一定相同。

### 算法的收敛性

&emsp;&emsp;主要是证明Novikoff定理，纯数学的东西，公式要码到吐了...不过找到了一篇笔记，分享给大家《[Convergence Proof for the Perceptron Algorithm](http://www.cs.columbia.edu/~mcollins/courses/6998-2012/notes/perc.converge.pdf)》

### 对偶形式

&emsp;&emsp;对偶形式的基本思想是，将$$w$$ 和$$b$$ 表示为实例$$x_i$$ 和标记$$y_i$$ 的线性组合形式，通过求解其系数而求得$$w$$ 和$$b$$ 。至于为什么...现在我还不太明白，也许以后学着学着就明白了吧。(书中提到与第七章支持向量机相对应，可能到那时候就明白了)

&emsp;&emsp;假设初始值$$w_0,b_0$$ 均为0，对误分类点$$(x_i,y_i)$$ 通过

$$w \leftarrow w+\eta y_ix_i$$ 

$$b \leftarrow b+\eta y_i$$ 

&emsp;&emsp;更新$$w,b$$ ，假设更新次数为n次，则$$w,b$$ 关于$$(x_i,y_i)$$ 的增量分别为$$ \alpha_iy_ix_i$$ 和$$ \alpha_iy_i$$ ，这里$$\alpha_i = n_i\eta$$ ，可以得到

$$w = \sum\limits_{i=1}^N \alpha_iy_ix_i$$ 

$$b = \sum\limits_{i=1}^N \alpha_iy_i$$ 

&emsp;&emsp;这里，$$\alpha \geq 0,i=1,2,...,N$$ ，当$$\eta = 1$$ 时，表示第$$i$$ 个样本由于被误实例而进行的更新次数。某实例更新次数越多，表示它距离超平面$$S$$ 越近，也就越难正确分类。换句话说，这样的实例对学习结果影响最大。

#### 算法2  (感知机学习算法的对偶形式)

输入: 训练数据集$$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_n,y_n) \right \}$$ , 其中$$x_i \in X = \mathbb{R}^n,y_i \in Y = \left \{ -1,+1 \right \},i = 1,2,...,N$$ ；学习率$$\eta(0<\eta\leq1)$$ ；

输出: $$\alpha,b$$ ；感知机模型$$f(x)=sign(\sum\limits_{j=1}^N \alpha_jy_jx_j\cdot x_i+b)$$ 

(1) $$\alpha \leftarrow 0,b\leftarrow 0$$ 

(2) 在训练集中选取数据$$(x_i,y_i)$$ 

(3) 如果$$y_i(\sum\limits_{j=1}^N \alpha_jy_jx_j\cdot x_i+b)\leq 0$$ 

$$\alpha \leftarrow \alpha_i+\eta$$

$$b\leftarrow b+\eta y_i$$ 

(4) 转至 (2) , 直到训练集中没有错误分类点。

&emsp;&emsp;由于训练实例仅以内积的形式出现，为方便，可预先将训练集中实例间的内积计算出来并以矩阵形式存储，这个矩阵就是Gram矩阵。

$$ G = \begin{bmatrix} x_i\cdot x_j\end{bmatrix}_{N\times N}$$ 

&emsp;&emsp;还是上面的例题，这次我们用对偶形式给出答案。

```python
import numpy as np

train = np.array([[[3, 3], 1], [[4, 3], 1], [[1, 1], -1]])
 
a = np.array([0, 0, 0])
b = 0
Gram = np.array([])
y = np.array(range(len(train))).reshape(1, 3)# 标签
x = np.array(range(len(train) * 2)).reshape(3, 2)# 特征
 
# 计算Gram矩阵
def gram():
    g = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(len(train)):
        for j in range(len(train)):
            g[i][j] = np.dot(train[i][0], train[j][0])
    return g
 
# 更新权重
def update(i):
    global a, b
    a[i] = a[i] + 1
    b = b + train[i][1]
    print(a, b) 

# 计算到超平面的距离
def cal(key):
    global a, b, x, y
    i = 0
    for data in train:
        y[0][i] = data[1]
        i = i + 1
    temp = a * y
    res = np.dot(temp, Gram[key])
    res = (res + b) * train[key][1]
    return res[0]
 
# 检查是否可以正确分类
def check():
    global a, b, x, y
    flag = False
    for i in range(len(train)):
        if cal(i) <= 0:
            flag = True
            update(i)
    if not flag:
        i = 0
        for data in train:
            y[0][i] = data[1]
            x[i] = data[0]
            i = i + 1
        temp = a * y
        w = np.dot(temp, x)
        print("The result: w: " + str(w) + ", b: "+ str(b))
        return False
    flag = False
     

Gram = gram()# 初始化Gram矩阵
for i in range(1000):
    check()
    if check() == False:
        break
```

&emsp;&emsp;我们得到如下结果，与之前得到的结果一样：

> ```
> [1 0 0] 1
> [1 0 1] 0
> [1 0 2] -1
> [1 0 3] -2
> [2 0 3] -1
> [2 0 4] -2
> [2 0 5] -3
> The result: w: [[1 1]], b: -3
> ```

## 小结

&emsp;&emsp;总算打完了，我机器学习直接是从Tensorflow搭建神经网络入的门，以前只是对其他机器学习算法有一些了解，这次真正学起来，感觉真不容易。这最简单的感知机是最简单的机器学习算法，也是神经网络的基础。

&emsp;&emsp;真心推荐李航老师的这本《统计学习方法》很棒，入门必备。

&emsp;&emsp;打LeTeX公式好累啊~~~!

## 参考文章

1. [《李航：统计学习方法》--- 感知机算法原理与实现](http://blog.csdn.net/u013358387/article/details/53303932)
2. [《统计学习方法》读书笔记——感知机](http://www.cnblogs.com/OldPanda/archive/2013/04/12/3017100.html)