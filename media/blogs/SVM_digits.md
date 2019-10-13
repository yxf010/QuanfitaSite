---
title: 机器学习入门实战——线性支持向量机实战digits数据集
categories: 
- note
tags: 
- Machine Learning
- SVM
- digits
copyright: true
mathjax: true
---



关于支持向量机的理论知识查看：[支持向量机]()

## digits数据集概述

digits.data：手写数字特征向量数据集，每一个元素都是一个64维的特征向量。

digits.target：特征向量对应的标记，每一个元素都是自然是0-9的数字。

digits.images：对应着data中的数据，每一个元素都是8*8的二维数组，其元素代表的是灰度值，转化为以为是便是特征向量。

## 代码实战

先导入数据，我们直接使用sklearn为我们准备好的数据集

```python
from sklearn.datasets import load_digits

digits = load_digits()
digits.data.shape
```

> (1797, 64)

将数据集进行划分

```python
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
```

数据标准化，导入线性支持向量机并训练

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train,Y_train)
Y_predict = lsvc.predict(X_test)
```

然后，我们对模型进行评估

```python
print('The Accuracy of Linear SVC is',lsvc.score(X_test,Y_test))
```



> The Accuracy of Linear SVC is 0.948888888889



```python
from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_predict,target_names=digits.target_names.astype(str)))
```

代码参考：《Python机器学习及实践：从零开始通往Kaggle竞赛之路》 























