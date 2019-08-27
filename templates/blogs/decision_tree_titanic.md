---
title: 机器学习入门实战——决策树算法实战Titanic数据集
categories: 
- note
tags: 
- Machine Learning
- Titanic
- decision tree
copyright: true
mathjax: true
---



关于决策树的理论知识可以查看：[决策树](http://quanfita.cn/2018/03/03/decision_tree/)

## Titanic数据集概述

&emsp;&emsp;RMS泰坦尼克号的沉没是历史上最臭名昭着的沉船之一。 1912年4月15日，在首航期间，泰坦尼克号撞上一座冰山后沉没，2224名乘客和机组人员中有1502人遇难。这一耸人听闻的悲剧震撼了国际社会，导致了更好的船舶安全条例。预测是否有乘客幸存下来的泰坦尼克号。

## 代码实战

首先，一如既往导入数据集，并查看一下部分数据

```python
import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.head()
```
![这里写图片描述](https://img-blog.csdn.net/2018032220364373?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwNjExNjAx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
info可以查看一下数据集的基本信息，我们可以看到数据集中有部分缺失值

```python
titanic.info()
```
```
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 1313 entries, 0 to 1312
> Data columns (total 11 columns):
> row.names    1313 non-null int64
> pclass       1313 non-null object
> survived     1313 non-null int64
> name         1313 non-null object
> age          633 non-null float64
> embarked     821 non-null object
> home.dest    754 non-null object
> room         77 non-null object
> ticket       69 non-null object
> boat         347 non-null object
> sex          1313 non-null object
> dtypes: float64(1), int64(2), object(8)
> memory usage: 112.9+ KB
```
我们从中选取三个特征来对数据进行预测

```python
X = titanic[['pclass','age','sex']]
y = titanic['survived']

X.info()
```
```
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 1313 entries, 0 to 1312
> Data columns (total 3 columns):
> pclass    1313 non-null object
> age       633 non-null float64
> sex       1313 non-null object
> dtypes: float64(1), object(2)
> memory usage: 30.9+ KB
```
age具有缺失值，所以我们要对缺失值进行处理

```python
X['age'].fillna(X['age'].mean(),inplace=True)
X.info()
```




```
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 1313 entries, 0 to 1312
> Data columns (total 3 columns):
> pclass    1313 non-null object
> age       1313 non-null float64
> sex       1313 non-null object
> dtypes: float64(1), object(2)
> memory usage: 30.9+ KB
```
之后就是将数据集划分成训练集和测试集

```python
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.25,random_state=33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)
X_test = vec.fit_transform(X_test.to_dict(orient='record'))

```

> ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']

导入决策树模型，并进行预测

```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
y_predict = dtc.predict(X_test)
```

最后，检查一下模型的效果

```python
from sklearn.metrics import classification_report

print(dtc.score(X_test,Y_test))
print(classification_report(y_predict,Y_test,target_names=['died','survived']))
```

![这里写图片描述](https://img-blog.csdn.net/2018032220361216?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMwNjExNjAx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
代码参考：《Python机器学习及实践：从零开始通往Kaggle竞赛之路》