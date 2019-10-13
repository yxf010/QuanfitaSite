---
title: 蓝桥杯日常刷题——练习1097：蛇行矩阵
categories: 
- 算法练习
tags: 
- 蓝桥杯
copyright: true
mathjax: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

**题目描述**

蛇形矩阵是由1开始的自然数依次排列成的一个矩阵上三角形。

**输入**

本题有多组数据，每组数据由一个正整数N组成。（N不大于100）

**输出**

对于每一组数据，输出一个N行的蛇形矩阵。两组输出之间不要额外的空行。矩阵三角中同一行的数字用一个空格分开。行尾不要多余的空格。

**样例输入**

```
5
```

**样例输出**

```
1 3 6 10 15
2 5 9 14
4 8 13
7 12
11
```

**题目分析**

解法一：

如果，我们只从数学规律角度去分析这道题，应该把这道问题分成行和列来分别计算

列的规律

```
1
1+1
1+1+2
1+1+2+3
...
```

我们可以写成公式$$1+\sum\limits_{i = 1}^{n}(i-1)$$ 

而行的规律

```
1 1+2 1+2+3 1+2+3+4 ...
2 2+3 2+3+4 
```

设行首数据为$$a_0$$ 写成公式$$\sum\limits_{i=1}^{n}(a_0+i-1)$$ 

**题目代码**

```c++
#include<cstdio>
using namespace std;


int main()
{
    int b;
    while(~scanf("%d",&b))
    {
        int a = 1,m=1,n=2,s=0;
        for(int j = 0;j<b;j++){
            m+=j;
            printf("%d ",m);
            for(int k = n ;k<=b;k++){
                s=s+k;
                printf("%d",m+s);
                if(k!=b) printf(" ");
            }
            s=0;
            n++;
            printf("\n");
        }
    }
    return 0;
}
```

解法二：

其实也不难发现，我们如果将第一行的数据存储一下，或许还可以使算法更高效

我们发现以下规律

```
a[0]	a[1]	a[2]	...
a[1]-1	a[2]-1	a[3]-1	...
...
```

**题目代码**

```c++
#include<cstdio>
using namespace std;
const int Max_N = 100;
int main()
{
    int a[Max_N],n;
    while(~scanf("%d",&n))
    {
        for(int i = 0;i<n;i++)
        {
            a[i] = (i+1)*(i+2)/2;
        }
        for(int j = 0;j<n;j++)
        {
            for(int k = j;k<n;k++)
            {
                printf("%d",a[k]);
                if(k!=n-1) printf(" ");
                else printf("\n");
                a[k] = a[k]-1;
            }
        }
    }
    return 0;
}
```

原题链接：[C语言网](http://www.dotcpp.com/oj/problem1097.html)

