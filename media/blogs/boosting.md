---
title: 机器学习入门之《统计学习方法》笔记整理——提升方法
categories: 
- note
tags: 
- Machine Learning
- Boosting
copyright: true
mathjax: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

&emsp;&emsp;提升方法的思路是综合多个分类器，得到更准确的分类结果。 说白了就是“三个臭皮匠顶个诸葛亮”。

##提升方法

### 提升方法AdaBoost算法

&emsp;&emsp;提升方法思路比较简单，它意在通过改变训练样本之间相对的权重，从而学习出多个分类器，并将这些分类器进行线性组合，从而提高分类的性能。
&emsp;&emsp;从以上思路可以看出，提升方法将一个复杂的任务分配给多个专家进行判断，并且对判断的结果进行综合，这样做要比只让一个专家来判断要好，毕竟大家说好才是真的好。

&emsp;&emsp;AdaBoost是提升算法的代表，所谓提升算法，指的是一种常用的统计学习方法，应用广泛且有效。在分类问题中，它通过改变训练样本的权重，学习多个分类器，并将这些分类器进行线性组合，提髙分类的性能。



#### 算法  (AdaBoost)

输入：训练数据集$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$ ，其中，$x_i\in X=\mathbb{R}^n$ ，$y_i\in Y=\{-1,+1\}$ ，弱学习算法；

输出：最终分类器$G(x)$ 。

(1) 初始化训练数据的权值分布

$$\begin{matrix}D_1=(w_{11},...,w_{1i},...,w_{1N}),& w_{1i}=\frac{1}{N}, & i=1,2...,N \end{matrix}$$ 

(2) 对$m=1,2,...,M$ 

&emsp;&emsp;(a) 使用具有权值分布$D_m$ 的训练数据集学习，得到基本分类器

$$G_m(x):X\rightarrow\{ -1,+1 \}$$ 

&emsp;&emsp;(b) 计算$G_m(x)$ 在训练数据集上的分类误差率

$$e_m=P(G_m(x_i)\neq y_i)=\sum\limits_{i=1}^{N}w_{m_i}I(G_m(x_i)\neq y_i)$$ 

&emsp;&emsp;(c) 计算$G_m(x)$ 的系数

$$\alpha_m=\frac{1}{2}\log\frac{1-e_m}{e_m}$$ 

这里的对数是自然对数。

&emsp;&emsp;(d) 更新训练数据集的权值分布

$$D_{m+1}=(w_{m+1,1},...,w_{m+1,i},...,w_{m+1,N})$$ 

$$\begin{matrix} w_{m+1,i}=\frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i)) & i=1,2,...,N \end{matrix}$$ 

也可以写成

$$w_{m+1,i}=\begin{cases} \frac{w_{mi}}{Z_m}e^{-\alpha_m},& G_m(x_i)=y_i \\ \frac{w_{mi}}{Z_m}e^{-\alpha_m},& G_m(x_i)\neq y_i \end{cases}$$ 

这里，$Z_m$ 是规范化因子

$$Z_m=\sum\limits_{i=1}^{N}w_{mi}\exp(-\alpha_my_iG_m(x_i))$$ 

它使$D_{m+1}$ 成为一个概率分布。

(3) 构建基本分类器的线性组合

$$f(x)=\sum\limits_{m=1}^{M}\alpha_mG_m(x)$$ 

得到最终分类器

$$G(x)=sign(f(x))=sign\left( \sum\limits_{m=1}^{M}\alpha_mG_m(x) \right)$$ 

&emsp;&emsp;从以上算法可以看到：最开始步骤1，我们假设了样本具有均匀的权重分布，它将产生一个基本分类器$G_1(x)$ 。步骤2是一个m从1到M的循环过程，每一次循环都可以产生一个弱分类器。

1. 分类误差率实际上就是被误分类点的权值之和。
2. 在计算当前弱分类器在线性组合中的系数时，当$e\geq0.5$ 时，$\alpha\geq0$，并且随着e的减小而增大，正好印证了需要使误差率小的弱分类器的权值更大这个事实。
3. 每一个样本的权值$w$ ，都与它原来的标签$$y_i$$ 以及预测的标签$G_m(x_i)$ 有关，当预测正确即它们同号时，exp指数是一个负值，这样就会减小原来样本点的权重；当预测不正确即它们异号时，exp指数是一个正值，它会增加当前样本点的权重。这正印证了我们需要使被误分类样本的权值更大这个事实。



### AdaBoost算法的解释

&emsp;&emsp;AdaBoost算法还有另一个解释，即可以认为AdaBoost算法是模型为加法模型、损失函数为指数函数、学习算法为前向分步算法时的二类分类学习方法。



#### 前向分步算法

&emsp;&emsp;考虑加法模型（additive model)

$$f(x)=\sum\limits_{m=1}^{M}\beta_mb(x;\gamma_m)$$ 

&emsp;&emsp;其中，$b(x;\gamma_m)$ 为基函数，$\gamma_m$ 为基函数的参数，$\beta_m$ 为基函数的系数。显然，$f(x)=\sum\limits_{m=1}^{M}\beta_mb(x;\gamma_m)$ 是一个加法模型。

&emsp;&emsp;在给定训练数据及损失函数的条件下，学习加法模型$f(x)$ 成为经验风险极小化即损失函数极小化问题：

$$\min\limits_{\beta_m,\gamma_m}\sum\limits_{i=1}^{N}L\left( y_i,\sum\limits_{m=1}^{M}\beta_mb(x_i;\gamma_m) \right)$$ 

&emsp;&emsp;通常这是一个复杂的优化问题。前向分步算法（forward stage wise algorithm)求解这一优化问题的想法是：因为学习的是加法模型，如果能够从前向后，每一步只学习一个基函数及其系数，逐步逼近优化目标函数式，那么就可以简化优化的复杂度。具体地，每步只需优化如下损失函数：

$$\min\limits_{\beta,\gamma}\sum\limits_{i=1}^{N}L\left( y_i,\beta b(x_i;\gamma) \right)$$ 



#### 算法  (前向分步算法)

输入：训练数据集$T=\{(x_{1},y_{1}),(x_2,y_2),...，(x_N,y_N)\}$ ，损失函数$L(y,f(x))$ 和基函数的集合$\{ b(x;\gamma) \}$ ；

输出：加法模型$f(x)$ .

(1) 初始化$f_0(x)=0$ 

(2) 对$m=1,2,...,M$ 

&emsp;&emsp;(a) 极小化损失函数

$$(\beta_m,\gamma_m)=\arg \min\limits_{\beta,\gamma} \sum\limits_{i=1}^{N}L(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma))$$ 

得到参数$\beta_m,\gamma_m$ 

&emsp;&emsp;(b) 更新

$$f_m(x)=f_{m-1}(x)+\beta_mb(x;\gamma_m)$$ 

(3) 得到加法模型

$$f(x)=f_M(x)=\sum\limits_{m=1}^{M} \beta_mb(x;\gamma_m)$$ 

&emsp;&emsp;这样，前向分步算法将同时求解从$m=1$ 到$M$ 所有参数$\beta_m,\gamma_m$ 的优化问题简化为逐次求解各个$\beta_m,\gamma_m$ 的优化问题。



#### 前向分步算法与AdaBoost

&emsp;&emsp;由前向分步算法可以推导出AdaBoost，AdaBoost算法是前向分歩加法算法的特例。这时，模型是由基本分类器组成的加法模型，损失函数是指数函数。



### 提升树

&emsp;&emsp;提升树是以分类树或回归树为基本分类器的提升方法。提升树被认为是统计学习中性能最好的方法之一。



#### 提升树模型

&emsp;&emsp;提升方法实际采用加法模型（即基函数的线性组合）与前向分步算法。以决策树为基函数的提升方法称为提升树（boosting tree)。对分类问题决策树是二叉分类树，对回归问题决策树是二叉回归树。在原著例题中看到的基本分类器，可以看作是由一个根结点直接连接两个叶结点的简单决策树，即所谓的决策树桩（decision stump)。提升树模型可以表示为决策树的加法模型：

$$f_M(x)=\sum\limits_{m=1}^{M} T(x;\Theta_m)$$ 

其中，$T(x;\Theta_m)$ 表示决策树；$\Theta_m$ 为决策树的参数；$M$ 为树的个数。



#### 提升树算法

&emsp;&emsp;提升树算法采用前向分步算法。首先确定初始提升树$f_m(x)=0$ ,第$m$ 歩的模型是

$$f_m(x)=f_{m-1}(x)+T(x;\Theta_m)$$ 

其中，$f_{m-1}(x)$ 为当前模型，通过经验风险极小化确定下一棵决策树的参数$\Theta_m$ 

$$\hat\Theta_m=\arg\min\limits_{\Theta_m} \sum\limits_{i=1}^{N}L(y_i,f_{m-1}(x_i)+T(x_i;\Theta_m))$$ 

&emsp;&emsp;由于树的线性组合可以很好地拟合训练数据，即使数据中的输入与输出之间的关系很复杂也是如此，所以提升树是一个髙功能的学习算法。

&emsp;&emsp;不同问题有大同小异的提升树学习算法，其主要区别在于使用的损失函数不同。包括用平方误差损失函数的回归问题，用指数损失函数的分类问题，以及用一般损失函数的一般决策问题。

&emsp;&emsp;对于二类分类问题，提升树算法只需将AdaBoost算法中的基本分类器限制为二类分类树即可，可以说这时的提升树算法是AdaBoost算法的特殊情况。



#### 算法  (回归问题的提升树算法)

输入：线性可分训练数据集$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$ ，其中，$x_i\in X\subseteq\mathbb{R}^n$ ，$y_i\in Y\subseteq \mathbb{R}$ ；

输出：提升树$f_M(x)$ .

(1) 初始化$f_0(x)=0$ 

(2) 对$m=1,2,...,M$ 

&emsp;&emsp;(a) 计算残差

$$\begin{matrix} r_{mi}=y_i-f_{m-1}(x_i) ,& i=1,2,...,N \end{matrix}$$ 

&emsp;&emsp;(b) 拟合残差$r_{mi}$ 学习一个回归树，得到$T(x;\Theta_m)$ 

&emsp;&emsp;(c) 更新$f_m(x)=f_{m-1}(x)+T(x;\Theta_m)$ 

(3) 得到回归问题提升树

$$f_M(x)=\sum\limits_{m=1}^{M} T(x;\Theta_m)$$ 



#### 算法  (梯度提升算法)

输入：线性可分训练数据集$T = \left \{ (x_1,y_1),(x_2,y_2),...,(x_N,y_N) \right \}$ ，其中，$x_i\in X\subseteq\mathbb{R}^n$ ，$y_i\in Y\subseteq \mathbb{R}$ ，损失函数$L(y,f(x))$ ；

输出：回归树$\hat f(x)$ .

(1) 初始化

$$f_0(x)=\arg\min\limits_c\sum\limits_{i=1}^{N}L(y_i,c)$$ 

(2) 对$m=1,2,...,M$ 

&emsp;&emsp;(a) 对$i=1,2,...,N$ ，计算

$$r_{mi}=-\left[\frac{\partial L(y_i,f(x_i)) } {\partial f(x_i)}\right]_{f(x)=f_{m-1}(x)}$$ 

&emsp;&emsp;(b) 对$r_{mi}$ 拟合一个回归树，得到第$m$ 棵树的叶节点区域$R_{mj},j=1,2,...,J$ 

&emsp;&emsp;(c) 对$j=1,2,...,J$ ，计算

$$c_{mj}=\arg\min\limits_c\sum\limits_{x_i\in R_{mj}}L(y_i,f_{m-1}(x_i)+c)$$ 

&emsp;&emsp;(d) 更新$f_m(x)=f_{m-1}(x)+\sum\limits_{j=1}^{J}c_{mj}I(x\in R_{mj})$ 

(3) 得到回归树

$$\hat f(x)=f_M(x)=\sum\limits_{m=1}^M\sum\limits_{j=1}^Jc_{mj}I(x\in R_{mj})$$ 



## 参考文章

1. [提升方法](http://www.hankcs.com/ml/adaboost.html)
2. [《统计学习方法（李航）》讲义 第08章 提升方法](https://www.cnblogs.com/itmorn/p/7751276.html)
3. [提升方法及AdaBoost](http://blog.csdn.net/wy250229163/article/details/53510015)