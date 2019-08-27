---
title: 机器学习入门实战——支持向量机实战Boston房价数据集
categories: 
- note
tags: 
- Machine Learning
- SVM
- Boston
copyright: true
mathjax: true
---



更多支持向量机的理论知识查看：[支持向量机]()

## 

该数据集来源于1978年美国某经济学杂志上。该数据集包含若干波士顿房屋的价格及其各项数据，每个数据项包含14个数据，分别是房屋均价及周边犯罪率、是否在河边等相关信息，其中最后一个数据是房屋均价。

## 代码实战

这里我们说简单点儿，我们从sklearn中的datasets中导入数据集，导入需要的库，将数据集进行划分，再标准化

```python
from sklearn.datasets import load_boston
boston = load_boston()

from sklearn.cross_validation import train_test_split
import numpy as np

X = boston.data
y = boston.target

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.25,random_state=33)

from sklearn.preprocessing import StandardScaler

ss_X = StandardScaler()
ss_Y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)
Y_train = ss_Y.fit_transform(Y_train)
Y_test = ss_Y.fit_transform(Y_test)
```

由于支持向量机的核函数我们可以自己选择，所以我们选择三种核函数进行对比

```python
from sklearn.svm import SVR
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,Y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train,Y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train,Y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)
```

我们来分别检测一下模型的性能

```python
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print("The value of R-squared of linear SVR is",linear_svr.score(X_test,Y_test))
print("The mean squared error of linear SVR is",mean_squared_error(ss_Y.inverse_transform(Y_test),ss_Y.inverse_transform(linear_svr_y_predict)))
print("The mean absolute error of linear SVR is",mean_absolute_error(ss_Y.inverse_transform(Y_test),ss_Y.inverse_transform(linear_svr_y_predict)))

print("The value of R-squared of poly SVR is",poly_svr.score(X_test,Y_test))
print("The mean squared error of poly SVR is",mean_squared_error(ss_Y.inverse_transform(Y_test),ss_Y.inverse_transform(poly_svr_y_predict)))
print("The mean absolute error of poly SVR is",mean_absolute_error(ss_Y.inverse_transform(Y_test),ss_Y.inverse_transform(poly_svr_y_predict)))

print("The value of R-squared of RBF SVR is",rbf_svr.score(X_test,Y_test))
print("The mean squared error of RBF SVR is",mean_squared_error(ss_Y.inverse_transform(Y_test),ss_Y.inverse_transform(rbf_svr_y_predict)))
print("The mean absolute error of RBF SVR is",mean_absolute_error(ss_Y.inverse_transform(Y_test),ss_Y.inverse_transform(rbf_svr_y_predict)))

```



> The value of R-squared of linear SVR is 0.654497663771
> The mean squared error of linear SVR is 26.7906984256
> The mean absolute error of linear SVR is 3.41002068375
> The value of R-squared of poly SVR is 0.23496198912
> The mean squared error of poly SVR is 59.3220377532
> The mean absolute error of poly SVR is 4.19595019294
> The value of R-squared of RBF SVR is 0.71072756206
> The mean squared error of RBF SVR is 22.4305593192
> The mean absolute error of RBF SVR is 2.81406224321

代码参考：《Python机器学习及实践：从零开始通往Kaggle竞赛之路》 








