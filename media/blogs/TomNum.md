---
title: 蓝桥杯日常刷题——练习1118：Tom数
categories: 
- 算法练习
tags: 
- 蓝桥杯
copyright: true
mathjax: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

**题目描述**

​	正整数的各位数字之和被Tom称为Tom数。求输入数（<2^32）的Tom数!

**输入**

​	每行一个整数(<2^32).

**输出**

​	每行一个输出,对应该数的各位数之和.

**样例输入**

```
12345
56123
82
```

**样例输出**

```
15
17
10
```

**题目分析**

这个题目考的是基本的数学知识，没有啥可分析的，直接看代码吧

**题目代码**

```c++
#include<cstdio>
using namespace std;

int Tom(long a)
{
    int sum=0;
    while(a)
    {
        sum+=a%10;
        a/=10;
    }
    return sum;
}

int main()
{
    long n;
    while(~scanf("%ld",&n))
    {
        printf("%d\n",Tom(n));
    }
    return 0;
}

```

