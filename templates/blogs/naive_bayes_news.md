---
title: 机器学习入门实战——朴素贝叶斯实战新闻组数据集
categories: 
- note
tags: 
- Machine Learning
- naive bayes
- 20newsgroups
copyright: true
mathjax: true
---



﻿## 朴素贝叶斯实战新闻组数据集

关于朴素贝叶斯的相关理论知识可查看：[朴素贝叶斯法](http://quanfita.cn/2018/02/21/naive_Bayes/)



### 关于新闻组数据集

20newsgroups数据集是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。一些新闻组的主题特别相似(e.g. comp.sys.ibm.pc.hardware/comp.sys.mac.hardware)，还有一些却完全不相关 (e.g misc.forsale /soc.religion.christian)。

20个新闻组数据集包含大约18000个新闻组，其中20个主题分成两个子集:一个用于训练(或开发)，另一个用于测试(或用于性能评估)。训练集和测试集之间的分割是基于特定日期之前和之后发布的消息。



### 代码实战

首先，还是导入数据集

```python
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])
```

我们这里打印出来一个新闻例子，如下

> 18846
> From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>
> Subject: Pens fans reactions
> Organization: Post Office, Carnegie Mellon, Pittsburgh, PA
> Lines: 12
> NNTP-Posting-Host: po4.andrew.cmu.edu
>
> 
>
> I am sure some bashers of Pens fans are pretty confused about the lack
> of any kind of posts about the recent Pens massacre of the Devils. Actually,
> I am  bit puzzled too and a bit relieved. However, I am going to put an end
> to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they
> are killing those Devils worse than I thought. Jagr just showed you why
> he is much better than his regular season stats. He is also a lot
> fo fun to watch in the playoffs. Bowman should let JAgr have a lot of
> fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final
> regular season game.          PENS RULE!!!
>

接下来，划分数据集，还是75%训练集，25%测试集

```python
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)
```

我们需要对文本特征进行提取，我们这里使用CountVectorizer来提取特征。CountVectorizer能够将文本词块化，通过计算词汇的数量来将文本转化成向量（更多文本特征提取内容可查看https://www.cnblogs.com/Haichao-Zhang/p/5220974.html）。然后我们导入模型来学习数据。

```python
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train,Y_train)
y_predict = mnb.predict(X_test)
```

最后，我们还是一样，检验一下模型的准确度

```python
from sklearn.metrics import classification_report
print('The Accuracy of Navie Bayes Classifier is',mnb.score(X_test,Y_test))
print(classification_report(Y_test,y_predict,target_names = news.target_names))
```
![这里写图片描述](http://img.blog.csdn.net/20180225212238191?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
代码参考：《Python机器学习及实践：从零开始通往Kaggle竞赛之路》