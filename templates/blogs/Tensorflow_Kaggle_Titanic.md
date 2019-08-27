---
title: Tensorflow简单神经网络解决Kaggle比赛Titanic问题
categories: 
- DeepLearning
tags: 
- TensorFlow
- Kaggle
- Titanic
copyright: true
---

&emsp;&emsp;又到了假期，忙碌了一个学期，终于可以休息一下了。

&emsp;&emsp;一直想再Kaggle上参加一次比赛，在学校要上课，还跟老师做个项目，现在有时间了，就马上用Kaggle的入门比赛试试手。

&emsp;&emsp;一场比赛，总的来说收获不小，平时学习的时候总是眼高手低，结果中间出现令人吐血的失误 >_< 

------

## Kaggle比赛介绍

![这里写图片描述](http://img.blog.csdn.net/20180131173522397?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

&emsp;&emsp;简而言之，Kaggle 是玩数据、ML 的开发者们展示功力、扬名立万的江湖，网址：https://www.kaggle.com/

&emsp;&emsp;Kaggle虽然高手云集，但是对于萌新们来说也是非常友好的，这次的Titanic问题就是适合萌新Getting Started的入门题。

> **Kaggle 是当今最大的数据科学家、机器学习开发者社区，其行业地位独一无二。**
>
> (此话引用自[谷歌收购 Kaggle 为什么会震动三界（AI、机器学习、数据科学界）？](https://www.leiphone.com/news/201703/ZjpnddCoUDr3Eh8c.html))




## Titanic问题概述

**Titanic: Machine Learning from Disaster**

![这里写图片描述](http://img.blog.csdn.net/20180131173704627?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**比赛说明**
&emsp;&emsp;RMS泰坦尼克号的沉没是历史上最臭名昭着的沉船之一。 1912年4月15日，在首航期间，泰坦尼克号撞上一座冰山后沉没，2224名乘客和机组人员中有1502人遇难。这一耸人听闻的悲剧震撼了国际社会，导致了更好的船舶安全条例。

&emsp;&emsp;沉船导致生命损失的原因之一是乘客和船员没有足够的救生艇。虽然幸存下来的运气有一些因素，但一些人比其他人更有可能生存，比如妇女，儿童和上层阶级。

&emsp;&emsp;在这个挑战中，我们要求你完成对什么样的人可能生存的分析。特别是，我们要求你运用机器学习的工具来预测哪些乘客幸存下来的悲剧。

**目标**
&emsp;&emsp;这是你的工作，以预测是否有乘客幸存下来的泰坦尼克号或不。
&emsp;&emsp;对于测试集中的每个PassengerId，您必须预测Survived变量的0或1值。

**度量值**
&emsp;&emsp;您的分数是您正确预测的乘客的百分比。这被称为“准确性”。

**提交文件格式**
&emsp;&emsp;你应该提交一个csv文件，正好有418个条目和一个标题行。如果您有额外的列（超出PassengerId和Survived）或行，您的提交将会显示错误。

该文件应该有2列：

&emsp;PassengerId（按任意顺序排序）
&emsp;生存（包含你的二元预测：1存活，0死亡）


![这里写图片描述](http://img.blog.csdn.net/20180131173736309?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 数据总览

&emsp;&emsp;首先，我们先把一些库和训练数据导入

```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf

train_data = pd.read_csv('train.csv')
print(train_data.info())
```

&emsp;&emsp;简单的看一下训练数据的信息，其中Embarked有两个缺失值，Age缺失值较多，Cabin有效值太少了跟本没什么用。

> ```
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 891 entries, 0 to 890
> Data columns (total 12 columns):
> PassengerId    891 non-null int64
> Survived       891 non-null int64
> Pclass         891 non-null int64
> Name           891 non-null object
> Sex            891 non-null object
> Age            714 non-null float64
> SibSp          891 non-null int64
> Parch          891 non-null int64
> Ticket         891 non-null object
> Fare           891 non-null float64
> Cabin          204 non-null object
> Embarked       889 non-null object
> dtypes: float64(2), int64(5), object(5)
> memory usage: 83.6+ KB
> None
> ```



## 数据清洗

&emsp;&emsp;在我们开始搭建神经网络进行训练之前，数据清洗是必要的。这一步可以简单一些，不过如果想要得到更好的效果，清洗之前的数据分析还是不可少的。这里的数据分析，我就不再赘述了，给大家推荐一篇博客，上面有很详细的分析过程——[Kaggle_Titanic生存预测](http://blog.csdn.net/Koala_Tree/article/details/78725881)

&emsp;&emsp;我们用随机森林算法，对'Age'的缺失值进行预测，当然这里也可以用其他回归算法，来进行预测

```python
from sklearn.ensemble import RandomForestRegressor
age = train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
age_notnull = age.loc[(train_data.Age.notnull())]
age_isnull = age.loc[(train_data.Age.isnull())]
X = age_notnull.values[:,1:]
Y = age_notnull.values[:,0]
rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
rfr.fit(X,Y)
predictAges = rfr.predict(age_isnull.values[:,1:])
train_data.loc[(train_data.Age.isnull()),'Age'] = predictAges
```

&emsp;&emsp;如果对上一步觉得太麻烦，或不喜欢的话，可以更简单一点，直接把缺失值都给0

```python
train_data = train_data.fillna(0) #缺失字段填0
```

&emsp;&emsp;然后，对于性别'Sex'，我们将其二值化'male'为0，'female'为1

```python
train_data.loc[train_data['Sex']=='male','Sex'] = 0
train_data.loc[train_data['Sex']=='female','Sex'] = 1
```

&emsp;&emsp;我们把'Embarked'也填补下缺失值，因为缺失值较少，所以我们直接给它填补上它的众数'S'，把'S'，'C'，'Q'定性转换为0,1,2，这样便于机器进行学习

```python
train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data.loc[train_data['Embarked'] == 'S','Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C','Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q','Embarked'] = 2
```

&emsp;&emsp;最后，把'Cabin'这个与生存关系不重要而且有效数据极少的标签丢掉，再加上一个'Deceased'，代表的是是否遇难，这一步很重要，很重要，很重要！我在做的时候没加这个，后面网络的y的标签我也只设了1，训练出的模型跟没训练一样，所有的都是0。发现的时候，死的心都有了╥﹏╥...（希望不会有初学者和我犯一样的错误 ToT ）

```python
train_data.drop(['Cabin'],axis=1,inplace=True)
train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)
```

&emsp;&emsp;然后，我们再查看一下数据信息

```python
train_data.info()
```

&emsp;&emsp;这次信息就整齐多了

> ```
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 891 entries, 0 to 890
> Data columns (total 12 columns):
> PassengerId    891 non-null int64
> Survived       891 non-null int64
> Pclass         891 non-null int64
> Name           891 non-null object
> Sex            891 non-null object
> Age            891 non-null float64
> SibSp          891 non-null int64
> Parch          891 non-null int64
> Ticket         891 non-null object
> Fare           891 non-null float64
> Embarked       891 non-null object
> Deceased       891 non-null int64
> dtypes: float64(2), int64(6), object(4)
> memory usage: 83.6+ KB
> ```



## 模型搭建

&emsp;&emsp;现在我们把数据的X，Y进行分离，这里我们只选取了6个标签作为X，如果想让结果尽可能准确，请读者自行完善。

```python
dataset_X = train_data[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dataset_Y = train_data[['Deceased','Survived']]
```

&emsp;&emsp;这里，我们进行训练集和验证集的划分，在训练过程中，我们可以更好的观察训练情况，避免过拟合

```python
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(dataset_X.as_matrix(),
                                                 dataset_Y.as_matrix(),
                                                test_size = 0.2,
                                                random_state = 42)
```

&emsp;&emsp;做完以上工作，我们就可以开始搭建神经网络了，这里，我搭建的是一个简单两层的神经网络，激活函数使用的是线性整流函数Relu，并使用了交叉验证和Adam优化器（也可以使用梯度下降进行优化），设置学习率为0.001

```python
x = tf.placeholder(tf.float32,shape = [None,6],name = 'input')
y = tf.placeholder(tf.float32,shape = [None,2],name = 'label')
weights1 = tf.Variable(tf.random_normal([6,6]),name = 'weights1')
bias1 = tf.Variable(tf.zeros([6]),name = 'bias1')
a = tf.nn.relu(tf.matmul(x,weights1) + bias1)
weights2 = tf.Variable(tf.random_normal([6,2]),name = 'weights2')
bias2 = tf.Variable(tf.zeros([2]),name = 'bias2')
z = tf.matmul(a,weights2) + bias2
y_pred = tf.nn.softmax(z)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=z))
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
acc_op = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
```

&emsp;&emsp;下面开始训练，训练之前我先定义了个Saver，epoch为30次

```python
# 存档入口
saver = tf.train.Saver()

# 在Saver声明之后定义的变量将不会被存储
# non_storable_variable = tf.Variable(777)

ckpt_dir = './ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt:
        print('Restoring from checkpoint: %s' % ckpt)
        saver.restore(sess, ckpt)
    
    for epoch in range(30):
        total_loss = 0.
        for i in range(len(X_train)):
            feed_dict = {x: [X_train[i]],y:[Y_train[i]]}
            _,loss = sess.run([train_op,cost],feed_dict=feed_dict)
            total_loss +=loss
        print('Epoch: %4d, total loss = %.12f' % (epoch,total_loss))
        if epoch % 10 == 0:
            accuracy = sess.run(acc_op,feed_dict={x:X_val,y:Y_val})
            print("Accuracy on validation set: %.9f" % accuracy)
            saver.save(sess, ckpt_dir + '/logistic.ckpt')
    print('training complete!')
    
    accuracy = sess.run(acc_op,feed_dict={x:X_val,y:Y_val})
    print("Accuracy on validation set: %.9f" % accuracy)
    pred = sess.run(y_pred,feed_dict={x:X_val})
    correct = np.equal(np.argmax(pred,1),np.argmax(Y_val,1))
    numpy_accuracy = np.mean(correct.astype(np.float32))
    print("Accuracy on validation set (numpy): %.9f" % numpy_accuracy)
    
    saver.save(sess, ckpt_dir + '/logistic.ckpt')
    
    '''
    测试数据的清洗和训练数据一样，两者可以共同完成
    '''
    
    # 读测试数据  
    test_data = pd.read_csv('test.csv')  
    
    #数据清洗, 数据预处理  
    test_data.loc[test_data['Sex']=='male','Sex'] = 0
    test_data.loc[test_data['Sex']=='female','Sex'] = 1 
    
    age = test_data[['Age','Sex','Parch','SibSp','Pclass']]
    age_notnull = age.loc[(test_data.Age.notnull())]
    age_isnull = age.loc[(test_data.Age.isnull())]
    X = age_notnull.values[:,1:]
    Y = age_notnull.values[:,0]
    rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    rfr.fit(X,Y)
    predictAges = rfr.predict(age_isnull.values[:,1:])
    test_data.loc[(test_data.Age.isnull()),'Age'] = predictAges
    
    test_data['Embarked'] = test_data['Embarked'].fillna('S')
    test_data.loc[test_data['Embarked'] == 'S','Embarked'] = 0
    test_data.loc[test_data['Embarked'] == 'C','Embarked'] = 1
    test_data.loc[test_data['Embarked'] == 'Q','Embarked'] = 2
    
    test_data.drop(['Cabin'],axis=1,inplace=True)
      
    #特征选择  
    X_test = test_data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]  
      
    #评估模型  
    predictions = np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1)  
    
    #保存结果  
    submission = pd.DataFrame({  
        "PassengerId": test_data["PassengerId"],  
        "Survived": predictions  
    })  
    submission.to_csv("titanic-submission.csv", index=False)  
```

&emsp;&emsp;我们把生成的提交文件在Kaggle官网上进行提交，Score为0.79425，效果还可以，不过还有很多需要改进的地方

![这里写图片描述](http://img.blog.csdn.net/20180131173808505?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

----------


## 参考文章

1. [Kaggle_Titanic生存预测 -- 详细流程吐血梳理](http://blog.csdn.net/Koala_Tree/article/details/78725881)
2. [谷歌收购 Kaggle 为什么会震动三界（AI、机器学习、数据科学界）？](https://www.leiphone.com/news/201703/ZjpnddCoUDr3Eh8c.html)
3. [《深度学习原理与TensorFlow实践》课程代码](https://github.com/DeepVisionTeam/TensorFlowBook/tree/master/Titanic)





