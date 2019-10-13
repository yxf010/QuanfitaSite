---
title: 机器学习入门之《统计学习方法》笔记整理——最大熵模型
categories: 
- note
tags: 
- Machine Learning
- 最大熵模型
copyright: true
mathjax: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

### 最大熵原理

&emsp;&emsp;最大熵原理是概率模型学习的一个准则，最大熵原理认为，学习概率模型时，在所有的可能的概率模型中，熵最大的模型是最好的模型。通常用约束条件来确定概率模型的集合，所以，熵最大原理也可以描述为在满足约束条件的模型集合中选取熵最大的模型。

首先回顾几个概念：

**熵**

&emsp;&emsp;假设离散随机变量$$X$$ 的概率分布是$$P(X)$$ ，则其熵为

$$H(P)=-\sum \limits_x P(x)\log P(x)$$ 

&emsp;&emsp;满足下列不等式:

$$0\leq H(P) \leq \log \left |X\right |$$ 

**联合熵和条件熵**

&emsp;&emsp;两个随机变量的$$X，Y$$ 的联合分布，可以形成联合熵，用$$H(X,Y)$$ 表示

条件熵$$H(X|Y) = H(X,Y) - H(Y)$$ 

$$H(X|Y) = H(X,Y) - H(Y)=-\sum \limits_{x,y} p(x,y)\log p(x|y)$$ 

**相对熵与互信息**

&emsp;&emsp;设$$p(x),q(x)$$ 是$$X$$ 中取值的两个概率分布，则$$p$$ 对$$q$$ 的相对熵是：

$$D(p||q)=\sum \limits_x p(x) \log \frac{p(x)}{q(x)}=E_{p(x)}\log \frac{p(x)}{q(x)}$$ 

&emsp;&emsp;两个随机变量$$X，Y$$ 的互信息，定义为$$X，Y$$ 的联合分布和独立分布乘积的相对熵。

$$I(X,Y)=D(P(X,Y)||P(X)P(Y))$$ 

$$I(X,Y)=\sum \limits_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$ 

直观讲， 最大熵原理认为要选择的概率模型

  （1）首先必须满足已有的事实，即约束条件。

  （2）在没有更多信息的情况下，就假设那些不确定的部分都是等可能的。

&emsp;&emsp;但是等可能不容易操作，而熵是一个可优化的数值指标。最大熵原理通过熵的最大化来表示等可能性。



### 最大熵模型的定义

&emsp;&emsp;最大熵模型假设分类模型是一个条件概率分布$$P(Y|X)$$ ,$$X$$ 为特征，$$Y$$ 为输出。

&emsp;&emsp;给定一个训练集$$T=\{(x_{1},y_{1}),(x_2,y_2),...，(x_N,y_N)\}$$ ,其中$$x$$ 为$$n$$ 维特征向量，$$y$$ 为类别输出。我们的目标就是用最大熵模型选择一个最好的分类类型。

&emsp;&emsp;在给定训练集的情况下，我们可以得到总体联合分布$$P(X,Y)$$ 的经验分布$$\tilde{P}(X,Y)$$ 和边缘分布$$P(X)$$ 的经验分布$$\tilde P(X)$$ 。$$\tilde{P}(X,Y)$$ 即为训练集中$$X,Y$$ 同时出现的次数除以样本总数$$N$$ ，$$\tilde P(X)$$ 即为训练集中$$X$$ 出现的次数除以样本总数$$N$$ 。

&emsp;&emsp;用特征函数$$f(x,y)$$ 描述输入$$x$$ 和输出$$y$$ 之间的关系。定义为：

$$f(x)=\begin{cases}1, & \text{ x与y满足某个关系 }  \\ 0, & \text{ 否则 } \end{cases}$$ 

&emsp;&emsp;可以认为只要出现在训练集中出现的$$(x_i,y_i)$$ ,其$$f(x_i,y_i)=1$$ . 同一个训练样本可以有多个约束特征函数。

&emsp;&emsp;特征函数$$f(x,y)$$ 关于经验分布$$\tilde{P}(X,Y)$$ 的期望值，用$$E_{\tilde P}(f)$$ 表示为:　

$$E_{\tilde P}(f)=\sum \limits_{x,y} \tilde P(x,y)f(x,y)$$

　　　

&emsp;&emsp;特征函数$$f(x,y)$$ 关于条件分布$$P(Y|X)$$ 和经验分布$$\tilde P(X)$$ 的期望值，用$$E_P(f)$$ 表示为:

$$E_P(f)=\sum \limits_{x,y}\tilde P(x)P(y|x)f(x,y)$$ 

&emsp;&emsp;如果模型可以从训练集中学习，我们就可以假设这两个期望相等。即：

$$E_{\tilde P}(f)=E_P(f)$$ 

&emsp;&emsp;上式是最大熵模型学习的约束条件，假如我们有n个特征函数$$f_i(x,y),i=1,2,...,n$$ 就有n个约束条件。

&emsp;&emsp;这样我们就得到了最大熵模型的定义如下：

&emsp;&emsp;假设满足所有约束条件的模型集合为：

$$E_{\tilde P}(f_i)=E_P(f_i),i=1,2,...,n$$ 

&emsp;&emsp;定义在条件概率分布$$P(Y|X)$$ 上的条件熵为：

$$H(P)=−\sum \limits_{x,y}\tilde P(x)P(y|x)\log P(y|x)$$ 

&emsp;&emsp;我们的目标是得到使$$H(P)$$ 最大的时候对应的$$P(y|x)$$ ，这里可以对$$H(P)$$ 加了个负号求极小值，这样做的目的是为了使$$−H(P)$$ 为凸函数，方便使用凸优化的方法来求极值。



### 最大熵模型的学习

&emsp;&emsp;对于给定的训练数据集$$T=\{（x_1，y_1）,（x_2，y_2）,(x_3，y_3),...,(x_n，y_n)\}$$ 以及特征函数$$f_i(x,y),i=1,2,3,...,n$$ ，最大熵模型的学习等价于约束的最优化问题：

$$\begin{matrix}  \max \limits_{P \in C} & H(P)=-\sum \limits_{x,y} \tilde P(x)P(y|x)\log P(y|x) \\ s.t. & E_p(f_i)=E_{\tilde P},i=1,2,...,n \\ & \sum \limits_y P(y|x) =1 \end{matrix} $$  

&emsp;&emsp;引入朗格朗日乘子$$w$$ ，定义拉格朗日函数$$L(P,w)$$ 

$$\begin{align*} L(P,w) &= -H(P)+w_0 \left ( 1-\sum \limits_y P(y|x) \right ) +\sum \limits_{i=1}^{n} w_i(E_{\tilde P}(f_i)-E_p(f_i)) \\  &=\sum \limits_{x,y} \tilde P(x)P(y|x)\log P(y|x)+w_0\left( 1-\sum \limits_{y} P(y|x) \right) \\  &+\sum \limits_{x,y}^{n}w_i \left( \sum \limits_{x,y} \tilde P(x,y)f_i(x,y)-\sum \limits_{x,y} \tilde P(x)P(y|x)f_i(x,y) \right ) \end{align*}$$ 

最优化的原始问题：

$$\min \limits_{P \in C} \max \limits_w L(P,w)$$ 

对偶问题是：

$$ \max \limits_w \min \limits_{P \in C} L(P,w)$$ 

&emsp;&emsp;由于L(P,W)是P的凸函数，原始问题的解与对偶问题的解是等价的。这里通过求对偶问题的解来求原始问题的解。

**第一步求解内部极小化问题**，记为：

$$\Psi(w)=\min \limits_{P \in C} L(P,w)=L(P_w,w)$$ 

通过微分求导，得出$$P$$ 的解是：

$$P_w(y|x)=\frac{1}{Z_w(x)}\exp \left( \sum \limits_{i=1}^{n}w_if_i(x,y) \right)$$ 

$$Z_w(x)=\sum \limits_y \exp \left ( \sum \limits_{i=1}^{n}w_if_i(x,y) \right)$$ 

**第二步求外部的极大化问题：**

$$\max \limits_w \Psi(w)$$ 

最后的解记为：

$$w^*=\arg \max \limits_w \Psi(w)$$ 

**第三步可以证明对偶函数的极大化等价于第一步求解出的P的极大似然估计**，所以将最大熵模型写成更一般的形式.

$$P_w(y|x)=\frac{1}{Z_w(x)}\exp \left(\sum \limits_{i=1}^{n}w_if_i(x,y)\right)$$ 

$$Z_w(x)=\sum \limits_y \exp \left ( \sum \limits_{i=1}^{n}w_if_i(x,y) \right)$$ 



### 模型学习的最优化算法

&emsp;&emsp;最大熵模型的学习最终可以归结为以最大熵模型似然函数为目标函数的优化问题。这时的目标函数是凸函数，因此有很多种方法都能保证找到全局最优解。例如改进的迭代尺度法(IIS)，梯度下降法，牛顿法或拟牛顿法，牛顿法或拟牛顿法一般收敛比较快。

#### 算法  (改进的迭代尺度算法IIS)

输入：特征函数$$f_1,f_2,...,f_n$$ ；经验分布$$\tilde P(X,Y)$$ ，模型$$P_w(y|x)$$ 

输出：最优参数值$$w_i^*$$ ，最优模型$$P_{w^*}$$ 

(1) 对所有$$i\in \{ 1,2,...,n \}$$ ，取初值$$w_i=0$$ 

(2) 对每一$$i\in\{ 1,2,...,n \}$$ ：

&emsp;&emsp;(a) 令$$\delta_i$$ 是方程

$$\sum \limits_{x,y} \tilde P(x)P(y|x)f_i(x,y)\exp (\delta_i \sum \limits_{i=1}^n f_i(x,y))=E_{\tilde P}(f_i)$$ 

&emsp;&emsp;(b) 更新$$w_i$$ ：$$w_i\leftarrow w_i + \delta_i$$ 

(3) 如果不是所有$$w_i$$ 都收敛，重复步(2).



#### 算法  (最大熵模型学习的BFGS算法)

输入：特征函数$$f_1,f_2,...,f_n$$ ；经验分布$$\tilde P(X,Y)$$ ，目标函数$$f(w)$$ ，梯度$$g(w)=\nabla f(w)$$ ，精度要求$$\varepsilon$$ ；

输出：最优参数值$$w^*$$ ，最优模型$$P_{w^*}(y|x)$$ 

(1) 选定初始点$$w^{(0)}$$ ，取$$B_0$$ 为正定对称矩阵，置$$k=0$$ 

(2) 计算$$g_k=g(w^{(k)})$$ .  若$$\left \| g_k \right \|<\varepsilon$$ ，则停止计算，得$$w^*=w^{(k)}$$ ；否则转(3)

(3) 由$$B_kp_k=-g_k$$ 求出$$p_k$$ 

(4) 一维搜索：求$$\lambda_k$$ 使得

$$f(w^{(k)}+\lambda_kp_k)=\min \limits_{\lambda \geq 0} f(w^{(k)}+\lambda p_k)$$ 

(5) 置$$w^{(k+1)}=w^{(k)}+\lambda_kp_k$$ 

(6) 计算$$g_{k+1}=g(w^{(k+1)})$$ ，若$$\left \| g_k \right \|<\varepsilon$$ ，则停止计算，得$$w^*=w^{(k)}$$ ；否则，按下式求出$$B_{k+1}$$ ：

$$B_{k+1}=B_{k}+\frac{y_ky_k^T}{y_k^T\delta_k}-\frac{B_k\delta_k\delta_k^T B_k}{\delta_k^TB_k\delta_k}$$ 

其中，

$$\begin{matrix} y_k=g_{k+1}-g_k, & \delta_k=w^{(k+1)}-w^{(k)} \end{matrix}$$ 

(7) 置$$k=k+1$$ ，转(3).



## 小结

**最大熵模型的优点：**

1. 最大熵统计模型获得的是所有满足约束条件的模型中信息熵极大的模型,作为经典的分类模型时准确率较高。
2.  可以灵活地设置约束条件，通过约束条件的多少可以调节模型对未知数据的适应度和对已知数据的拟合程度

**最大熵模型的缺点：**

1. 由于约束函数数量和样本数目有关系，导致迭代过程计算量巨大，实际应用比较难。



## 参考文章

1. [一步一步理解最大熵模型](https://www.cnblogs.com/wxquare/p/5858008.html)
2. [最大熵模型原理小结](https://www.cnblogs.com/pinard/p/6093948.html)