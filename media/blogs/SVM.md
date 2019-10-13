---
title: 机器学习入门之《统计学习方法》笔记整理——支持向量机
categories: 
- note
tags: 
- Machine Learning
- SVM
copyright: true
mathjax: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

﻿## 支持向量机

&emsp;&emsp;支持向量机(support vector machines, SVM)是一种二类分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器；支持向量机还包括核技巧，这使它成为实质上的非线性分类器。支持向量机的学习策略就是**间隔最大化**，可形式化为一个求解凸二次规划(convex quadratic programming)的问题，也等价于正则化的合页损失函数的最小化问。支持向量机的学习算法是**求解凸二次规划的最优化算法。**

&emsp;&emsp;支持向量机，其含义是通过**支持向量**运算的分类器。

&emsp;&emsp;**支持向量：**在求解的过程中，会发现只根据部分数据就可以确定分类器，这些数据称为支持向量。

### 线性可分支持向量机

#### 线性可分支持向量机

&emsp;&emsp;支持向量机的输入空间和特征空间是不同的，输入空间为欧氏空间或离散集合，特征空间是欧氏空间或希尔伯特空间。希尔伯特空间其实就可以当做欧氏空间的扩展，其空间维度可以是任意维的，包括无穷维，并且具有欧氏空间不具备的完备性。

&emsp;&emsp;这时，我们需要先回忆一下[感知机](http://blog.csdn.net/qq_30611601/article/details/79313609) ，因为这两个的决策函数是类似的：

&emsp;&emsp;给定线性可分训练数据集，通过间隔最大化或等价地求解相应的凸二次规划问题学习得到的分离超平面为： 

$$w^∗\cdot x+b^∗=0$$ 

&emsp;&emsp;以及相应的分类决策函数：

$$f(x)=sign(w^∗\cdot x+b^∗)$$ 

称为线性可分支持向量机。
![这里写图片描述](http://img.blog.csdn.net/20180308194708686?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

&emsp;&emsp;感知机通过训练一个超平面将平面或空间线性可分的点进行划分。 线性可分支持向量机也是如此，通过找寻分割平面来划分数据集。两者的区别，感知机的学习策略是误分类点到超平面距离和最小化，而线性可分支持向量机是基于硬间隔最大化的。



#### 函数间隔与几何间隔

**&emsp;&emsp;函数间隔：**对于给定的训练数据集T和超平面$(w, b)$ ，定义超平面关于样本点$(x_i, y_i)$ 的函数间隔为

$$\hat\gamma_i=y_i(w\cdot x_i+b)$$ 

&emsp;&emsp;定义超平面$(w,b)$ 关于训练数据集T的函数间隔为超平面$(w,b)$ 关于T中所有样本点$(x_i, y_i)$ 的函数间隔之最小值，即

$$\hat\gamma=\min\limits_{i=1,2,...,N} \hat\gamma_i$$ 

&emsp;&emsp;函数间隔可以表示分类预测的正确性及确信度。但是成比例地改变$w$ 和$b$ ，例如将它们改为$2w$ 和$2b$ ，超平面并没有改变，但函数间隔却成为原来的2倍。

![这里写图片描述](http://img.blog.csdn.net/20180308194835385?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
&emsp;&emsp;对分离超平面的法向量、加某些约束，如规范化，$\left \|w\right \|=1$ ，使得间隔是确定的。这时函数间隔成为几何间隔。

**&emsp;&emsp;几何间隔：**对于给定的训练数据集T和超平面$(w, b)$ ，定义超平面关于样本点$(x_i, y_i)$ 的函数间隔为

$$\gamma_i=y_i\left( \frac{w}{\left\| w \right \|}\cdot x_i+\frac{b}{\left\| w \right \|} \right)$$ 

&emsp;&emsp;定义超平面$(w,b)$ 关于训练数据集T的函数间隔为超平面$(w,b)$ 关于T中所有样本点$(x_i, y_i)$ 的函数间隔之最小值，即

$$\gamma=\min\limits_{i=1,2,...,N}\gamma_i$$ 

&emsp;&emsp;函数间隔和几何间隔的关系:

$$\gamma_i=\frac{\hat\gamma_i}{\left\| w \right \|}$$ 

$$\gamma=\frac{\hat\gamma}{\left\| w \right \|}$$ 

&emsp;&emsp;如果超平面参数$w$ 和$b$ 成比例地改变(超平面没有改变)，函数间隔也按此比例改变，而几何间隔不变。



#### 间隔最大化

&emsp;&emsp;支持向量机学习的基本想法是求解能够正确划分训练数据集并且几何间隔最大的分离超平面。对线性可分的训练数据集而言，线性可分分离超平面有无穷多个(等价于感知机)，但是几何间隔最大的分离超平面是唯一的。这里的间隔最大化又称为硬间隔最大化。   

&emsp;&emsp;间隔最大化的直观解释是：对训练数据集找到几何间隔最大的超平面意味着以充分大的确信度对训练数据进行分类，也就是说，不仅将正负实例点分开，而且对最难分的实例点(离超平面最近的点)也有足够大的确信度将它们分开。

&emsp;&emsp;这个问题可以表示为下面的约束最优化问题：

$$\begin{matrix} \max\limits_{w,b} & \gamma \\s.t. & y_i\left( \frac{w}{\left\| w \right\|}\cdot x_i+\frac{b}{\left\| w \right\|} \right) \geq \gamma, & i=1,2,...,N \end{matrix}$$ 

&emsp;&emsp;即：

$$\begin{matrix} \max\limits_{w,b} & \frac{\hat\gamma}{\left\| w \right\|} \\s.t. & y_i\left( w\cdot x_i+b \right) \geq \hat\gamma, & i=1,2,...,N \end{matrix}$$ 

&emsp;&emsp;由于$$\hat\gamma$$ 的取值并不影响最优化，所以这里我们为了计算方便取$$\hat\gamma=1$$ .目标函数变为：

$$\max\limits_{w,b} \frac{1}{\left\| w \right\|}$$ 

&emsp;&emsp;因为最大化$\frac{1}{\left\| w \right\|}$ 等价于最小化$\frac{1}{2}\left\| w \right\|^2$ (为什么？因为要将目标函数转换为一个凸二次规划问题，从而满足后面求对偶问题需要的KKT条件(什么是KKT条件？维基百科：[KKT条件](http://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions))，而且使所求的解为全局最优解。系数加个1/2是为了求导的时候约去系数，计算方便。)，从而将问题改写成：

$$\begin{matrix} \max\limits_{w,b} & \frac{1}{2}\left\|w\right\|^2 \\s.t. & y_i\left( w\cdot x_i+b \right)-1 \geq0, & i=1,2,...,N \end{matrix}$$ 



#### 算法  (线性可分支持向量机学习算法——最大间隔法)

输入：线性可分训练数据集$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$ ，其中，$x_i\in X=\mathbb{R}^n$ ，$y_i\in Y=\{-1,+1\}$ ，$i=1,2,...,N$ ；

输出：最大间隔分离超平面和分类决策函数。

(1) 构造并求解约束最优化问题：

$$\begin{matrix} \max\limits_{w,b} & \frac{1}{2}\left\|w\right\|^2 \\s.t. & y_i\left( w\cdot x_i+b \right)-1 \geq0, & i=1,2,...,N \end{matrix}$$ 

求得最优解$w^*,b^*$ .

(2) 由此得到分离超平面：

$$w^*\cdot x+b^*=0$$ 

分类决策函数

$$f(x)=sign(w^*\cdot x+b^*)$$ 



#### 学习的对偶算法

&emsp;&emsp;构建拉格朗日函数(Lagrange function)，引进拉格朗日乘子(Lagrange multiplier)：

$$L(w,b,\alpha)=\frac{1}{2}\left\|w\right\|^2-\sum\limits_{i=1}^{N} \alpha_iy_i(w\cdot x_i+b)+\sum\limits_{i=1}^{N}\alpha_i$$ 

&emsp;&emsp;根据拉格朗日对偶性，原始问题的对偶问题是拉格朗日函数的极大极小问题

$$\max\limits_{\alpha}\min\limits_{w,b}L(w,b,\alpha)$$ 

&emsp;&emsp;设$a^*$是对偶最优化问题的解，则存在下标$j$ 使得$a_j^* >0$ ，并可按下式求得原始最优化问题的解:

$$w^*=\sum\limits_{i=1}^{N}\alpha_i^*y_ix_i$$ 

$$b^*=y_i-\sum\limits_{i=1}^{N}\alpha_i^*y_i(x_i\cdot x_j)$$ 



#### 算法  (线性可分支持向量机学习算法)

输入：线性可分训练数据集$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$ ，其中，$x_i\in X=\mathbb{R}^n$ ，$y_i\in Y=\{-1,+1\}$ ，$i=1,2,...,N$ ；

输出：分离超平面和分类决策函数。

(1) 构造并求解约束最优化问题：

$$\min\limits_{\alpha} \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} \alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum\limits_{i=1}^{N}\alpha_i$$ 

$$\begin{matrix}s.t. & \sum\limits_{i=1}^{N}\alpha_iy_i=0\end{matrix}$$ 

$$\alpha_i\geq0,i=1,2,...,N$$ 

求得最优解$$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$$ .

(2) 计算

$$w^*=\sum\limits_{i=1}^{N}\alpha_i^*y_ix_i$$ 

并选择$\alpha^*$ 的一个正分量$\alpha_j^*>0$ ，计算

$$b^*=y_i-\sum\limits_{i=1}^{N}\alpha_i^*y_i(x_i\cdot x_j)$$ 

(3) 求得分离超平面

$$w^*\cdot x+b^*=0$$ 

分类决策函数：

$$f(x)=sign(w^*\cdot x+b^*)$$ 



### 线性支持向量机

&emsp;&emsp;上面所说的线性可分支持向量机是基于训练样本线性可分的理想状态，当训练样本中存在噪声或者特异点而造成线性不可分时，就需要用到线性支持向量机。 

&emsp;&emsp;在线性可分支持向量机中，我们假设函数间隔$\hat\gamma$ 为1，若存在噪声或特异点函数间隔处于 $(0,1)$ 中间，那么这些点就不满足问题的约束条件，也就线性不可分。为了解决这样的问题，引入了松弛变量$\xi_i\geq0$ ，使得函数间隔与松弛变量的和大于等于1，从而约束条件变为：

$$y_i(w\cdot x_i+b)\geq1-\xi_i$$ 

&emsp;&emsp;同时因为约束条件引入了$\xi_i$ ，所以目标函数也要改变，改为： 

$$\frac{1}{2}\left\| w \right\|^2+C\sum\limits_{i=1}^N \xi_i$$ 

&emsp;&emsp;这里，$C>0$ 称为惩罚参数，由问题决定。

&emsp;&emsp;依然构造拉格朗日函数，并转换为对偶问题： 

$$\begin{matrix}\min\limits_{\alpha} &  \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} \alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum\limits_{i=1}^{N}\alpha_i \end{matrix}$$ 

$$\begin{matrix}s.t. & \sum\limits_{i=1}^{N}\alpha_iy_i=0\end{matrix}$$ 

$$0\leq\alpha_i\leq C,i=1,2,...,N$$ 

&emsp;&emsp;其拉格朗日函数是

$$L(w,b,\xi,\alpha,\mu)\equiv \frac{1}{2}\left\| w \right\|^2+C\sum\limits_{i=1}^N \xi_i -\sum\limits_{i=1}^{N} \alpha_i(y_i(w\cdot x_i+b)-1+\xi_i)-\sum\limits_{i=1}^{N}\mu_i\xi_i$$ 

其中，$\alpha_i\geq0,\mu_i\geq0$ .



#### 算法  (线性支持向量机学习算法)

输入：线性可分训练数据集$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$ ，其中，$x_i\in X=\mathbb{R}^n$ ，$y_i\in Y=\{-1,+1\}$ ，$i=1,2,...,N$ ；

输出：分离超平面和分类决策函数。

(1) 选择惩罚参数$C>0$ ，构造并求解凸二次规划问题：

$$\begin{matrix}\min\limits_{\alpha} &  \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} \alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum\limits_{i=1}^{N}\alpha_i \end{matrix}$$ 

$$\begin{matrix}s.t. & \sum\limits_{i=1}^{N}\alpha_iy_i=0\end{matrix}$$ 

$$0\leq\alpha_i\leq C,i=1,2,...,N$$ 

求得最优解$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$ .

(2) 计算

$$w^*=\sum\limits_{i=1}^{N}\alpha_i^*y_ix_i$$ 

并选择$\alpha^*$ 的一个分量$\alpha_j^*$ 适合条件$0<\alpha_j^*<C$ ，计算

$$b^*=y_i-\sum\limits_{i=1}^{N}y_i\alpha_i^*(x_i\cdot x_j)$$ 

(3) 求得分离超平面

$$w^*\cdot x+b^*=0$$ 

分类决策函数：

$$f(x)=sign(w^*\cdot x+b^*)$$ 



#### 支持向量

&emsp;&emsp;支持向量有两种解释，一种是直观的解释，一种与对偶最优化问题的解$\alpha^*$ 联系起来。

##### 1. 支持向量和间隔边界

&emsp;&emsp;在线性可分情况下，训练数据集的样本点中与分离超平面跄离最近的样本点的实例称为**支持向量( support vector )。**支持向量是使约束条件式等号成立的点，即

$$y_i(w\cdot x_i+b)-1=0$$ 

对$y_i=+1$ 的正例点，支持向量在超平面$H_1:w\cdot x+b=1$ 

对$y_i=-1$ 的负例点，支持向量在超平面$H_2:w\cdot x+b=-1$ 



&emsp;&emsp;$H_1$ 和$H_2$ 之间的距离称为间隔(margin)。间隔依赖于分离超平面的法向量$w$ ，等于$\frac{2}{\left\|w\right\|}$ 。$H_1$ 和$H_2$ 称为间隔边界。

&emsp;&emsp;在决定分离超平面时只有支持向量起作用，而其他实例点并不起作用。如果移动支持向量将改变所求的解；但是如果移动其他实例点，甚至去掉这些点，则解是不会改变的。由于支持向量在确定分离超平面中起决定性作用，所以将这种分类模型称为支持向量机。支持向量的个数一般很少，所以支持向量机由很少的“重要的”训练样本确定。

##### 2. 支持向量和对偶最优化问题的解$α^∗$

&emsp;&emsp;在线性可分支持向量机中，$(w^∗,b^∗)$只依赖于训练数据中对应于$α^∗_i>0$的样本点$(x_i,y_i)$ ，而其他样本点对$(w^∗,b^∗)$ 没有影响，将训练数据中对应于$α^∗_i>0$ 的实例点$(x_i,y_i)$ 称为支持向量。 

&emsp;&emsp;支持向量一定在间隔边界上，由KKT互补条件可知: 

$$α^∗_i(y_i(w^∗⋅x_i+b^∗)−1)=0,i=1,2,⋯,N$$ 

&emsp;&emsp;对应于$α^∗_i>0$ 的实例点$(x_i,y_i)$ ，则有： 

$$y_i(w^∗⋅x_i+b^∗)−1=0$$ 

&emsp;&emsp;即$(x_i,y_i)$ 一定在间隔边界上，和前面的的支持向量定义是一致的。 

&emsp;&emsp;同时可以得出，非支持向量对应的$α^∗_i=0$，因为$y_i(w^∗⋅x_i+b^∗)−1>0$ ，故$α^∗_i=0$ 。



#### 合页损失函数

&emsp;&emsp;线性支持向量机学习还有另外一种解释，就是最小化以下目标函数

$$\sum\limits_{i=1}^{N}\left [1-y_i(w\cdot x_i+b) \right]_++\lambda\left\|w \right\|^2$$ 

&emsp;&emsp;目标函数的第1项是经验损失或经验风险，函数

$$L(y(w\cdot x+b))=\left[1-y(w\cdot x+b) \right]_+$$ 

称为合页损失函数.下标“+”表示下取正值的函数

$$\left [ z \right ]_+=\begin{cases}z, & \text{  } z>0 \\ 0 & \text{  } z\leq0\end{cases}$$ 

&emsp;&emsp;合页损失函数不仅要分类正确，而且确信度足够高时损失才是0。



### 非线性支持向量机

#### 核技巧

非线性分类问题：如果能用$\mathbb{R}^n$ 中的一个超曲面将正负例正确分开，则称这个问题为非线性可分问题.

求解方法：进行非线性变换，将非线性问题变成线性问题。

&emsp;&emsp;学习是隐式地在特征空间进行的，不需要显式地定义特征空间和映射函数。这样的技巧称为**核技巧**。

&emsp;&emsp;核技巧应用到支持向量机，其基本想法就是通过一个非线性变换将输入空间(欧氏空间$\mathbb{R}^n$ 或离散集合)对应于一个特征空间(希尔伯特空间$H$ )，使得在输入空间$\mathbb{R}^n$ 中的超曲面模型对应于特征空间$H$ 中的超平面模型(支持向量机)。

&emsp;&emsp;设$X$ 是输入空间，$H$ 为特征空间，如果存在一个映射映射函数

$$\phi(x):X\rightarrow H$$ 

&emsp;&emsp;使得对所有$x,z\in X$ ，函数$K(x,z)$ 满足条件

$$K(x,z)=\phi(x)\cdot \phi(z)$$ 

**则称$K(x,z)$ 为核函数。**

&emsp;&emsp;核技巧的想法是，在学习与预测中只定义核函数$K(x,z)$ ，而不显式地定义映射函数。对于给定的核$K(x,z)$ ，特征空间x和映射函数的取法并不唯一，可以取不同的特征空间，即便是在同一特征空间里也可以取不同的映射。

&emsp;&emsp;在对偶问题的目标函数中的内积$(x_i\cdot x_j)$ 可以用核函数$K(x_i, x_j)$ 来代替：

$$w(\alpha)= \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} \alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum\limits_{i=1}^{N}\alpha_i $$ 

&emsp;&emsp;分类决策函数也可用核函数代替，变为：

$$f(x)=sign\left( \sum\limits_{i=1}^{N_s}a_i^*y_i\phi(x_i)\cdot\phi(x)+b^* \right)=sign\left( \sum\limits_{i=1}^{n_s}a_i^*y_iK(x_i,x)+b^* \right)$$ 

&emsp;&emsp;这等价于经过映射函数将原来的输入空间变换到一个新的特征空间，将输入空间中的内积$x_i\cdot x_j$ 变换为特征空间中的内积$\phi(x_i)\cdot \phi(x_j)$ .

&emsp;&emsp;在新的特征空间里从训练样本中学习线性支持向量机。当映射函数是非线性函数时，学习到的含有核函数的支持向量机是非线性分类模型。

&emsp;&emsp;在核函数给定的条件下，可以利用解线性分类问题的方法求解非线性分类问题的支持向量机。

&emsp;&emsp;这里给出判定正定核的充要条件：

&emsp;&emsp;设$Κ：X\times X→\mathbb{R}$ 是对称函数，则$Κ(x,z)$ 为正定核函数的充要条件是对任意$x_i∈X$ ,$i=1,2,…,m$,$Κ(x,z)$ 对应的Gram矩阵： 

$$K=\left [ K(x_i,x_j) \right]_{m\times n}$$ 

是半正定矩阵。

&emsp;&emsp;由充要条件可以给出判定正定核的等价定义： 

&emsp;&emsp;设$X$ 为输入空间，$Κ(x,z)$ 是定义在$X\times X$ 对称函数，如果对任意$x_i∈X$ , $i=1,2,…,m$ , $Κ(x,z)$ 对应的Gram矩阵：

$$K=\left [ K(x_i,x_j) \right]_{m\times n}$$ 

是半正定矩阵，则称$Κ(x,z)$ 是正定核。 符合这样条件的函数，我们称它为正定核函数。 

**注意**：也有的核函数是非正定核，如多元二次核函数$K(x,z)=(\left\| x-z \right\|^2+c^2)^{\frac{1}{2}}$ 

&emsp;&emsp;在实际应用中，还经常用到Mercer定理还确定核函数。由Mercer定理得到的核函数称为Mercer核，正定核比Mercer核更具有一般性，因为正定核要求函数为定义空间上的对称函数，而Mercer核要求函数为对称连续函数。



#### 常用核函数

##### 1. 线性核函数

&emsp;&emsp;线性核函数是最简单的核函数，是径向基核函数的一个特例，公式为：

$$K(x,z)=x^Ty+c$$ 

&emsp;&emsp;主要用于线性可分的情形，在原始空间中寻找最优线性分类器，具有参数少速度快的优势。 

##### 2. 多项式核函数

&emsp;&emsp;多项式核适合于正交归一化数据，公式为：

$$K(x,z)=(x\cdot z+1)^p$$ 

&emsp;&emsp;多项式核函数属于全局核函数，允许相距很远的数据点对核函数的值有影响。参数$p$ 越大，映射的维度越高，计算量就会越大。当$p$ 过大时，学习复杂性也会过高，易出现过拟合。 

##### 3. 径向基核函数

&emsp;&emsp;径向基核函数属于局部核函数，当数据点距离中心点变远时，取值会变小，公式为：

$$K(x,z)=\exp(-\gamma\left\| x-z \right\|^2)$$ 

##### 4. 高斯核函数

&emsp;&emsp;高斯核函数可以看作是径向基核函数的另一种形式：

$$K(x,z)=\exp\left(-\frac{\left\| x-z \right\|^2}{2\sigma^2}\right)$$ 

&emsp;&emsp;高斯(径向基)核对数据中存在的噪声有着较好的抗干扰能力，由于其很强的局部性，其参数决定了函数作用范围，随着参数$\sigma$ 的增大而减弱。

##### 5. 字符串核函数

&emsp;&emsp;核函数不仅可以定义在欧氏空间上，还可以定义在离散数据的集合上。字符串核函数是定义在字符串集合上的核函数，可以直观地理解为度量一对字符串的相似度，在文本分类、信息检索等方面都有应用。

$$k_n(s,t)=\sum\limits_{u\in\Sigma^n}[\phi_n(s)]_u[\phi_n(t)]_u=\sum\limits_{u\in\Sigma^n}\sum\limits_{(i,j):s(i)=t(j)=u}\lambda^{l(i)}\lambda^{l(j)}$$ 

&emsp;&emsp;字符串核函数$k_n(s,t)$ 给出了字符串$s$ 和$t$ 中长度等于$n$ 的所有子串组成的特征向量的余弦相似度。两个字符串相同的子串越多，他们就越相似，字符串核函数的值就越大。字符串核函数可以由动态规划快速的计算。

##### 6. Sigmoid核函数

&emsp;&emsp;Sigmoid核函数来源于神经网络，被广泛用于深度学习和机器学习中，公式为： 

$$K(x,z)=\tanh(x\cdot z+c)$$ 

&emsp;&emsp;采用Sigmoid函数作为核函数时，支持向量机实现的就是一种多层感知器神经网络。支持向量机的理论基础（凸二次规划）决定了它最终求得的为全局最优值而不是局部最优值，也保证了它对未知样本的良好泛化能力。 



#### 算法  (非线性支持向量机学习算法)

输入：线性可分训练数据集$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$ ，其中，$x_i\in X=\mathbb{R}^n$ ，$y_i\in Y=\{-1,+1\}$ ，$i=1,2,...,N$ ；

输出：分类决策函数。

(1) 选择惩罚参数$C>0$ ，构造并求解凸二次规划问题：

$$\begin{matrix}\min\limits_{\alpha} &  \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N} \alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum\limits_{i=1}^{N}\alpha_i \end{matrix}$$ 

$$\begin{matrix}s.t. & \sum\limits_{i=1}^{N}\alpha_iy_i=0\end{matrix}$$ 

$$0\leq\alpha_i\leq C,i=1,2,...,N$$ 

求得最优解$\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)^T$ .

(2) 选择$\alpha^*$ 的一个分量$\alpha_j^*$ 适合条件$0<\alpha_j^*<C$ ，计算

$$b^*=y_i-\sum\limits_{i=1}^{N}y_i\alpha_i^*K(x_i, x_j)$$ 

(3) 分类决策函数：

$$f(x)=sign\left( \sum\limits_{i=1}^{n_s}a_i^*y_iK(x_i,x)+b^* \right)$$ 



### SMO算法

&emsp;&emsp;SMO算法是一种启发式算法，其基本思路是：如果所有变量的解都满足此最优化问题的KKT条件(Karush-Kuhn-Tucker conditions)，那么这个最优化问题的解就得到了。因为KKT条件是该最优化问题的充分必要条件。否则，选择两个变量，固定其他变量，针对这两个变量构建一个二次规划问题。这个二次规划问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变得更小。重要的是，这时子问题可以通过解析方法求解，这样就可以大大提高整个算法的计算速度。子问题有两个变量，一个是违反KKT条件最严重的那一个，另一个由约束条件自动确定。如此，SMO算法将原问题不断分解为子问题并对子问题求解，进而达到求解原问题的目的。

概要：SMO方法的中心思想是每次取一对$α_i$ 和$α_j$ ，调整这两个值。

参数：训练数据/分类数据/$C$ /$\xi$ /最大迭代数

过程：

> 初始化$\alpha$为0；
>
> 在每次迭代中 （小于等于最大迭代数），
>
> - 找到第一个不满足KKT条件的训练数据，对应的$α_i$，
>
> - 在其它不满足KKT条件的训练数据中，找到误差最大的x，对应的index的$α_j$ ，
>
> - $α_i$ 和$α_j$ 组成了一对，根据约束条件调整$α_i$ , $α_j$ 。

&emsp;&emsp;整个SMO算法包括两个部分:求解两个变量二次规划的解析方法和选择变量的启发式方法.



#### 坐标上升法

&emsp;&emsp;假设有优化问题： 

$$\begin{matrix} \max\limits_{\alpha}&W(\alpha_1,\alpha_2,...,\alpha_m) \end{matrix}$$ 

&emsp;&emsp;W是α向量的函数。利用坐标上升法（求目标函数的最小时即为坐标下降法）求解问题最优的过程如下： 

> //循环到函数收敛
>
> Loop until convergence{
>
> ​	//依次选取一个变量，将其作为固定值
>
> ​	For i = 1,2,...,m{
> ​		$$\alpha_i = \arg \max \limits_{\alpha_i} W(\alpha_1,\alpha_2,...,\alpha_i,...,\alpha_m)$$ 
> ​	}
> }

&emsp;&emsp;算法的思想为：每次只考虑一个变量进行优化，将其他变量固定。这时整个函数可以看作只关于该变量的函数，可以对其直接求导计算。然后继续求其他分变量的值，整个内循环下来就得到了$\alpha$ 的一组值，若该组值满足条件，即为我们求的值，否则继续迭代计算直至收敛。



#### SMO算法

&emsp;&emsp;参考坐标上升法，我们选择向量$\alpha$ 的一个变量，将其他变量固定进行优化，该处优化问题包含了约束条件， 变量必须满足等式约束$\sum\limits_{i=1}^{N}\alpha_iy_i=0$ ，所以每次选择两个变量进行优化

&emsp;&emsp;不失一般性，将设选择的两个变量为$\alpha_1，\alpha_2$，其他变量$\alpha_i (i=3,4,…,N)$ 是固定的。 于是优化问题的子问题可以写作： 

$$\begin{matrix}\min\limits_{\alpha_1,\alpha_2} & W(\alpha_1,\alpha_2)=\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2-(\alpha_1+\alpha_2) \\&+y_1\alpha_1\sum\limits_{i=3}^N y_i\alpha_iK_{i1}+y_2\alpha_2\sum\limits_{i=3}^{N}y_i\alpha_iK_i2\\s.t. & \alpha_1y_1+\alpha_2y_2=-\sum\limits_{i=3}^{N}y_i\alpha_i=\varsigma \\ & 0\leq\alpha_i\leq C,i=1,2 \end{matrix}$$ 

其中，$K_{ij}=K(x_i,x_j),i,j=1,2,...,N$ ，$\varsigma$ 是常数。



&emsp;&emsp;现在的问题就是如何选择两个变量构造最优子问题。SMO采用启发式选择方法选择变量。所谓启发式，即每次选择拉格朗日乘子时，优先选择前面样本系数中满足条件$0<α_i < C$ 的 $α_i$ 作优化，不考虑约束条件中相等的情况是因为在界上的样例对应的系数$α_i$  一般都不会改变。 

&emsp;&emsp;通过启发式搜索找到第一个变量，因为要考虑算法的收敛性，第二个变量显然不是随便选的。实际上，只要选择的两个变量中有一个违背KKT条件，那么目标函数在一步迭代后值就会减小,并且我们希望找到的$α_2$ 在更新后能够有足够大的变化。 



#### 算法  (SMO算法)

输入：线性可分训练数据集$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$ ，其中，$x_i\in X=\mathbb{R}^n$ ，$y_i\in Y=\{-1,+1\}$ ，$i=1,2,...,N$ ，精度$\varepsilon$ ；

输出：近似解$\hat\alpha$。

(1) 取初值$\alpha^{(0)}=0$，令$k$ ；

(2) 选取优化变量$\alpha_1^{(k)},\alpha_2^{(k)}$ ，解析求解两个变量的最优化问题，求得最优解$\alpha_1^{(k+1)},\alpha_2^{(k+1)}$ ，更新$\alpha$ 为$\alpha^{(k+1)}$ ；

(3) 若精度$\varepsilon$ 范围内满足停机条件

$$\sum\limits_{i=1}^{N}\alpha_iy_i=0$$  

$$0\leq\alpha_i\leq C,i=1,2,...,N$$ 

$$y_i\cdot g(x_i)=\begin{cases} \geq1,& \{x_i|\alpha_i=0\} \\=1,& \{x_i|0<\alpha_i<C\} \\  \leq1,& \{x_i|\alpha_i=C\}  \end{cases}$$ 

其中，

$$g(x_i)=\sum\limits_{j=1}^N \alpha_jy_jK(x_j,x_i)+b$$ 

则转(4) ；否则令$k=k+1$ ，转(2) ；

(4) 取$\hat\alpha=\alpha^{(k+1)}$ 



## 小结

&emsp;&emsp;力推一篇文章，写的太好了，我写的只能算是笔记，这才是真正详细地讲述原理——[支持向量机通俗导论（理解SVM的三层境界）](http://blog.csdn.net/macyang/article/details/38782399)



## 参考文章

1. [理解支持向量机](http://blog.csdn.net/shijing_0214/article/details/50982602)
2. [理解支持向量机（二）核函数](http://blog.csdn.net/shijing_0214/article/details/51000845)
3. [统计学习方法 李航---第7章 支持向量机](https://www.cnblogs.com/YongSun/p/4767130.html)
4. [理解数学空间，从距离到希尔伯特空间](http://blog.csdn.net/shijing_0214/article/details/51052208)
5. [机器学习实战 - 读书笔记(06) – SVM支持向量机](https://www.cnblogs.com/steven-yang/p/5658362.html)
6. [支持向量机(SVM)是什么意思？](https://www.zhihu.com/question/21094489)
7. [支持向量机通俗导论（理解SVM的三层境界）](http://blog.csdn.net/macyang/article/details/38782399)

