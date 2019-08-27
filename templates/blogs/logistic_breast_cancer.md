---
title: 机器学习入门实战——逻辑斯谛回归实战breast cancer数据集
categories: 
- note
tags: 
- Machine Learning
- logistic
- breast cancer
copyright: true
mathjax: true
---



更多有关逻辑斯谛回归的理论知识查看：[逻辑斯谛回归](http://quanfita.cn/2018/03/03/logistic_regression/)



## 代码实战

首先，我们还是先将需要用到的库导入，应为此数据集缺少名称，所以，使用pandas导入数据时，我们需要手动添加名称

```python
import pandas as pd
import numpy as np
import tensorflow as tf
column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size',
                'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size',
               'Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv('breast-cancer-train.csv',names=column_names)
data.head()
```



为了更好的了解数据集的情况，我们查看一下数据信息

```python
data.info()
data = data.replace(to_replace='?',value=np.nan)
data = data.dropna(how='any')
data.shape
```

> ```
> <class 'pandas.core.frame.DataFrame'>
> Int64Index: 683 entries, 0 to 698
> Data columns (total 11 columns):
> Sample code number             683 non-null int64
> Clump Thickness                683 non-null int64
> Uniformity of Cell Size        683 non-null int64
> Uniformity of Cell Shape       683 non-null int64
> Marginal Adhesion              683 non-null int64
> Single Epithelial Cell Size    683 non-null int64
> Bare Nuclei                    683 non-null object
> Bland Chromatin                683 non-null int64
> Normal Nucleoli                683 non-null int64
> Mitoses                        683 non-null int64
> Class                          683 non-null int64
> dtypes: int64(10), object(1)
> memory usage: 84.0+ KB
> ```

接下来，我们将数据集划分为训练集和测试集

```python
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)

```

将数据标准化，导入逻辑斯谛回归模型，然后就可以进行预测了

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

print(X_train.shape,Y_train.shape)

lr = LogisticRegression()
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)
```

最后，我们查看一下模型的效果

```python
from sklearn.metrics import classification_report
print(lr.score(x_test,y_test))
```

> 0.988304093567

代码参考：《Python机器学习及实践：从零开始通往Kaggle竞赛之路》





