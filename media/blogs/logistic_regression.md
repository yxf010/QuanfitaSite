<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
&emsp;&emsp;逻辑回归(logistic regression)是统计学习中的经典分类方法。其多用在二分类{0,1}问题上。最大嫡是概率模型学习的一个准则将其推广到分类问题得到最大熵模型(maximum entropy model)。逻辑回归模型与最大熵模型都属于对数线性模型。

##逻辑斯谛回归模型

###逻辑斯谛分布

&emsp;&emsp;设X是连续随机变量，\\(X\\)服从逻辑斯谛分布是指\\(X\\)具有下列分布函数和密度函数

$$F(x)=P(X\leq x)=\frac{1}{1+e^{-(x-\mu)/\gamma}}$$ 

$$f(x)=F^{'}(x)=\frac{e^{-(x-\mu)/\gamma}}{\gamma(1+e^{-(x-\mu)/\gamma})^2}$$ 



&emsp;&emsp;分布函数属于逻辑斯谛函数，其图形是一条S形曲线。

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTgwMzAzMjMyMzA0Mjc3)

### 二项逻辑斯谛回归模型

&emsp;&emsp;二项逻辑回归模型(binomial logistic regression model)是一种分类模型，用于二类分类。由条件概率分布\\(P(Y|X)\\)表示，形式为参数化的逻辑分布。这里，随机变量X取值为实数，随机变量Y取值为1或0。

&emsp;&emsp;二项逻辑斯谛回归模型是如下条件概率分布:

$$P(Y=1|x)=\frac{\exp(w\cdot x+b)}{1+\exp(w\cdot x+b)}$$ 

$$P(Y=0|x)=\frac{1}{1+\exp(w\cdot x+b)}$$ 

&emsp;&emsp;其中，\\(x\in\mathbb{R}^n,Y\in \{ 0,1 \}\\)，\\(w\in\mathbb{R}^n\\)和\\(b\in\mathbb{R}\\).

&emsp;&emsp;逻辑回归对线性回归经行了归一化操作，将输出范围规定在{0,1}。

&emsp;&emsp;几率，指一件事件发生的概率与不发生的概率的比值，那么事件的对数几率或logit函数是

$$logit(p)=\log \frac{p}{1-p}$$ 

&emsp;&emsp;因此，

$$logit(p)=\log \frac{P(Y=1|x)}{1-P(Y=1|x)}=w\cdot x$$ 

&emsp;&emsp;这就是说，在逻辑回归模型中，输出Y=1的对数几率是输入x的线性函数。

&emsp;&emsp;通过逻辑回归模型可以将线性函数转化为概率：

$$P(Y=1|x)=\frac{\exp(w\cdot x)}{1+\exp(w\cdot x)}$$ 

&emsp;&emsp;线性函数值越接近正无穷，概率越接近1；线性函数值越接近负无穷，概率值越接近0。这样的模型称为逻辑回归模型。



Python代码如下：

```python
import numpy as np
def predict(x,w):
    return 1.0/1.0+np.e**(-x.dot(w))

def iter_w(x, y, a, w):
    prediction = predict(x,w)
    g = (prediction - y) * x
    w = w+ a * g * (1.0 / y.size)
    return w

while counter < max_epochs:
    counter += 1
    for i in range(len(Y)):
        w = update(X[i,:], Y[i], a, w)
```



###模型参数估计

&emsp;&emsp;逻辑斯谛回归模型学习时，可以利用极大似然估计法估计模型参数

&emsp;&emsp;似然函数：

$$\prod \limits_{i=1}^{N} P(Y=1|x_i)^{y_i}(1-P(Y=1|x_i))^{1-y_i}$$ 

&emsp;&emsp;对数似然函数：

$$\begin{align\*} &L(w)=\sum \limits_{i=1}^{N}\left [ y_i\log P(Y=1|x_i)+(1-y_i)\log (1-P(Y=1|x_i)) \right ] \\\\  &=\sum \limits_{i=1}^{N}\left [ y_i\log \frac{P(Y=1|x_i)}{1-P(Y=1|x_i)}+\log (1-P(Y=1|x_i)) \right ] \\\\  &= \sum \limits_{i=1}^{N}\left [ y_i(w\cdot x_i)-\log (1+\exp(w\cdot x_i)) \right ]\end{align\*}$$ 

&emsp;&emsp;这样子，我们有了损失函数，这里我们只要将该函数极大化即可，求其最大值时的\\(w\\)即可。

&emsp;&emsp;具体过程可参考Andrew Ng的[CS229讲义](http://cs229.stanford.edu/notes/cs229-notes1.pdf), Part 2 logistic regression部分,[翻译](https://millearninggroup.github.io/Stanford-CS229-CN/translation/cs229-notes1-cn-2/)) 



###多项逻辑斯谛回归

&emsp;&emsp;多项逻辑斯谛回归用于多分类问题，其模型为

$$P(Y=k|x)=\frac{\exp(w_k\cdot x)}{1+\sum \limits_{k=1}^{K-1} \exp(w_k\cdot x)},k=1,2,...,K-1$$ 

$$P(Y=K|x)=\frac{1}{1+\sum \limits_{k=1}^{K-1} \exp(w_k\cdot x)}$$ 



##参考文章

1. [《统计学习方法》第六章逻辑斯蒂回归与最大熵模型学习笔记](http://blog.csdn.net/wjlucc/article/details/69264144?ref=myread)
2. [统计学习方法 李航---第6章 逻辑回归与最大熵模型](http://blog.csdn.net/demon7639/article/details/51011417)


