---
title: 蓝桥杯日常刷题——历届试题1434：回文数字
categories: 
- 算法练习
tags: 
- 蓝桥杯
copyright: true
mathjax: true
---



<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

**题目描述**

观察数字：12321，123321  都有一个共同的特征，无论从左到右读还是从右向左读，都是相同的。这样的数字叫做：回文数字。 
本题要求你找到一些5位或6位的十进制数字。满足如下要求： 
该数字的各个数位之和等于输入的整数。 

**输入**

一个正整数  n  (10< n< 100),  表示要求满足的数位和。

**输出**

若干行，每行包含一个满足要求的5位或6位整数。 
数字按从小到大的顺序排列。 
如果没有满足条件的，输出：-1 

**样例输入**

```
44 
```

**样例输出**

```
99899
499994
589985
598895
679976
688886
697796
769967
778877
787787
796697
859958
868868
877778
886688
895598
949949
958859
967769
976679
985589
994499
```

**题目分析**

解法一：

暴力破解法

既然题目中说了是五位和六位的数字，那么我们可以直接遍历从10000到999999的所有数字，从中进行筛选

**题目代码**

```c++
#include<iostream>
using namespace std;
int n;
bool flag = false;
bool huiwen(long a)
{
    long temp = a;
    long b = 0;
    while(temp){
        b = b*10;
        b += temp % 10;
        temp /= 10;
    }
    return a == b;
}
bool xiangjia(long a)
{
    long sum = 0;
    while(a){
        sum += a%10;
        a /= 10;
    }
    if(sum == n){
        return true;
    }
    return false;
}
int main()
{

    cin >> n;
    for(long i = 10000; i < 1000000; i++){
        if(huiwen(i)){
            if(xiangjia(i))
            {
                cout << i << endl;
                flag = true;
            }
        }
    }
    if(!flag)
        cout << "-1" << endl;

    return 0;
}
```

解法二：

暴力破解法虽然简单但是遍历10000到999999之间的所有数浪费了很多时间，所以为了降低运行时间，我们将每一位单独讨论首尾相等我们用一个变量保存，这样下来，我们只需遍历9^3次。

**题目代码**

```c++
#include<iostream>
using namespace std;
bool flag = false;
void solve(int x,int y,int z,int n)
{
    int a[3] = {0,0,0};
    for(a[0] = 1 ;a[0]<=9;a[0]++)
        for(a[1] = 0 ;a[1]<=9;a[1]++)
            for(a[2] = 0 ;a[2]<=9;a[2]++)
                if(z == 1){
                    if(x*a[0]+y*a[1]+z*a[2] == n){
                        cout<<a[0]<<a[1]<<a[2]<<a[1]<<a[0]<<endl;
                        flag = true;
                    }
                }
                else{
                    if(x*a[0]+y*a[1]+z*a[2] == n){
                        cout<<a[0]<<a[1]<<a[2]<<a[2]<<a[1]<<a[0]<<endl;
                        flag = true;
                    }
                }
}

int main()
{
    int n = 0;
    cin>>n;
    solve(2,2,1,n);
    solve(2,2,2,n);
    if(flag == false) cout<<"-1"<<endl;
    return 0;
}
```

原题链接：[C语言网](http://www.dotcpp.com/oj/problem1434.html)

