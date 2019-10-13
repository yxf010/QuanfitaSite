---
title: 数据结构详解——线性表（C++实现）
categories: 
- Data Structures
tags: 
- Linear List
copyright: true
mathjax: true
---



#线性表

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

&emsp;&emsp;线性表是最常用且是最简单的一种数据结构。形如：A1、A2、A3….An这样含有有限的数据序列，我们就称之为线性表。



##一、线性表的定义

**线性表：零个或多个数据元素的有限序列。**

线性表、包括顺序表和链表
顺序表（其实就是数组）里面元素的地址是连续的，
链表里面节点的地址不是连续的，是通过指针连起来的。



##二、线性表的抽象数据类型

&emsp;&emsp;线性表的抽象数据类型定义如下：

> ```
> ADT 线性表(List)
> Data
>     线性表的数据对象集合为{a1,a2,....,an},每个元素的类型均为DataType。其中，除了第一个元素a1外，每一个元素有且只有一个直接前驱元素，除最后一个元素an外，每一个元素有且只有一个直接后继元素。数据元素之间的关系是一对一的关系。
>
> Operation
>     InitList(*L):初始化操作，建立一个空的线性表。
>     ListEmpty(L):若线性表为空，返回true，否则返回false。
>     ClearList(*L):线性表清空。
>     GetElem(L,i,*e):将线性表L中第i个位置元素返回给e。
>     LocateElem(L,e):在线性表L中查找与给定值e相等的元素，如果查找成功,返回该元素在表中的序列号；否则，返回0表示失败。
>     ListInsert(*L,i,e):在线性表的第i个位置插入元素e。
>     ListDelete(*L,i,*e):删除线性表L中的第i个元素，并用e返回其值
>     ListLength(L):返回线性表L的元素个数。
>     PrintList(L):打印线性表
>     
> 对于不同的应用，线性表的基本操作是不同的，上述操作是最基本的。
> 对于实际问题中涉及的关于线性表的更复杂操作，完全可以用这些基本操作的组合来实现。
> ```



## 三、线性表的顺序存储

### 1. 顺序存储定义

&emsp;&emsp;顺序表，一般使用数组实现，事实上就是在内存中找个初始地址，然后通过占位的形式，把一定连续的内存空间给占了，然后把相同数据类型的数据元素依次放在这块空地中，数组大小有两种方式指定，一是静态分配，二是动态扩展。

&emsp;&emsp;顺序表相关的操作跟数组有关，一般都是移动数组元素。

![这里写图片描述](http://img.blog.csdn.net/20180311152120741?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 2. 顺序存储的实现方式

**结构**

&emsp;&emsp;我们直接来看顺序表的模板类的代码：

```C
const int MaxSize = 100;
template <class DataType>
class SeqList
{
public:
    SeqList(){length=0;}            //无参数构造方法
    SeqList(DataType a[],int n);    //有参数构造方法
    ~SeqList(){}                    //析构函数
    int Length(){return length;}    //线性表长度
    DataType Get(int i);            //按位查找
    int Locate(DataType x);         //按值查找
    void Insert(int i,DataType x);  //插入
    DataType Delete(int i);         //删除
    void PrintList();               //遍历
private:
    DataType data[MaxSize];         //顺序表使用数组实现
    int length;                     //存储顺序表的长度
};
```

顺序表的封装需要三个属性：

1. 存储空间的起始位置。数组data的存储位置就是线性表存储空间的存储位置
2. 线性表的最大存储容量。数组长度MAXSIZE
3. 线性表的当前长度。length

**注意**：数组的长度与线性表的当前长度是不一样的。数组的长度是存放线性表的存储空间的总长度，一般初始化后不变。而线性表的当前长度是线性表中元素的个数，是会改变的。

&emsp;&emsp;下面我们将实现顺序表的各个功能：

**有参数构造**：

&emsp;&emsp;创建一个长度为n的顺序表，需要将给定的数组元素作为线性表的数据元素传入顺序表中，并将传入的元素个数作为顺序表的长度

![这里写图片描述](http://img.blog.csdn.net/20180311152302345?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template <class DataType>
SeqList<DataType>::SeqList(DataType a[],int n)
{
    if(n>MaxSize) throw "wrong parameter";
    for(int i=0;i<n;i++)
        data[i]=a[i];
    length=n;
}
```

**按位查找**

&emsp;&emsp;按位查找的时间复杂度为$$O(1)$$ 。

```c++
template <class DataType>
DataType SeqList<DataType>::Get(int i)
{
    if(i<1 && i>length) throw "wrong Location";
    else return data[i-1];
}
```

**按值查找**

&emsp;&emsp;按值查找，需要对顺序表中的元素依次进行比较。

```c++
template <class DataType>
int SeqList<DataType>::Locate(DataType x)
{
    for(int i=0;i<length;i++)
        if(data[i]==x) return i+1;
    return 0;
}
```

**插入**

&emsp;&emsp;插入的过程中需要注意元素移动的方向，必须从最后一个元素开始移动，如果表满了，则引发上溢；如果插入位置不合理，则引发位置异常。

![这里写图片描述](http://img.blog.csdn.net/20180311152401708?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template <class DataType>
void SeqList<DataType>::Insert(int i,DataType x)
{
    if(length>=MaxSize) throw "Overflow";
    if(i<1 || i>length+1) throw "Location";
    for(int j=length;j>=i;j--)
        data[j]=data[j-1];
    data[i-1]=x;
    length++;
}
```

**删除**

&emsp;&emsp;注意算法中元素移动方向，移动元素之前必须取出被删的元素，如果表为空则发生下溢，如果删除位置不合理，则引发删除位置异常。

![这里写图片描述](http://img.blog.csdn.net/20180311152453294?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template <class DataType>
DataType SeqList<DataType>::Delete(int i)
{
    int x;
    if(length==0) throw "Underflow";
    if(i<1 || i>length) throw "Location";
    x = data[i-1];
    for(int j=i;j<length;j++)
        data[j-1] = data[j];
    length--;
    return x;
}
```

**遍历**

&emsp;&emsp;按下标依次输出各元素

```c++
template <class DataType>
void SeqList<DataType>::PrintList()
{
    for(int i=0;i<length;i++)
        cout<<data[i]<<endl;
}
```

&emsp;&emsp;完整代码示例(更多数据结构完整示例可见[GitHub](https://github.com/Quanfita/Data_Stuctures))：

```c++
#include<iostream>
using namespace std;

const int MaxSize = 100;
template <class DataType>
class SeqList
{
public:
    SeqList(){length=0;}            
    SeqList(DataType a[],int n);    
    ~SeqList(){}                    
    int Length(){return length;}    
    DataType Get(int i);            
    int Locate(DataType x);         
    void Insert(int i,DataType x);  
    DataType Delete(int i);         
    void PrintList();               
private:
    DataType data[MaxSize];         
    int length;                     
};

template <class DataType>
SeqList<DataType>::SeqList(DataType a[],int n)
{
    if(n>MaxSize) throw "wrong parameter";
    for(int i=0;i<n;i++)
        data[i]=a[i];
    length=n;
}

template <class DataType>
DataType SeqList<DataType>::Get(int i)
{
    if(i<1 && i>length) throw "wrong Location";
    else return data[i-1];
}

template <class DataType>
int SeqList<DataType>::Locate(DataType x)
{
    for(int i=0;i<length;i++)
        if(data[i]==x) return i+1;
    return 0;
}

template <class DataType>
void SeqList<DataType>::Insert(int i,DataType x)
{
    if(length>=MaxSize) throw "Overflow";
    if(i<1 || i>length+1) throw "Location";
    for(int j=length;j>=i;j--)
        data[j]=data[j-1];
    data[i-1]=x;
    length++;
}

template <class DataType>
DataType SeqList<DataType>::Delete(int i)
{
    int x;
    if(length==0) throw "Underflow";
    if(i<1 || i>length) throw "Location";
    x = data[i-1];
    for(int j=i;j<length;j++)
        data[j-1] = data[j];
    length--;
    return x;
}

template <class DataType>
void SeqList<DataType>::PrintList()
{
    for(int i=0;i<length;i++)
        cout<<data[i]<<endl;
}

int main()
{
    SeqList<int> p;
    p.Insert(1,5);
    p.Insert(2,9);
    p.PrintList();
    p.Insert(2,3);
    cout<<p.Length()<<endl;
    p.PrintList();
    cout<<p.Get(3)<<endl;
    p.Delete(2);
    p.PrintList();
    return 0;
}
```





### 3. 顺序存储的优缺点

**优点：**

- 随机访问特性，查找O(1)时间，存储密度高；
- 逻辑上相邻的元素，物理上也相邻；
- 无须为表中元素之间的逻辑关系而增加额外的存储空间；

**缺点：**

- 插入和删除需移动大量元素；
- 当线性表长度变化较大时，难以确定存储空间的容量；
- 造成存储空间的“碎片”





## 四、线性表的链式存储

### 1. 链式存储定义

&emsp;&emsp;线性表的链式存储结构的特点是用一组任意的存储单元存储线性表的数据元素，这组存储单元可以是连续的，也可以是不连续的。这就意味着，这些元素可以存在内存未被占用的任意位置。

![这里写图片描述](http://img.blog.csdn.net/20180311152556473?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

&emsp;&emsp;链表的定义是递归的，它或者为空null，或者指向另一个节点node的引用，这个节点含有下一个节点或链表的引用，线性链表的最后一个结点指针为“空”（通常用NULL或“^”符号表示）。

![这里写图片描述](http://img.blog.csdn.net/20180311152641656?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 2. 链式存储的实现方式

**存储方法**

```c
template<class DataType>
struct Node
{
	DataType data;				//存储数据
	Node<DataType> *next;		//存储下一个结点的地址
};
```

&emsp;&emsp;结点由存放数据元素的数据域和存放后继结点地址的指针域组成。 

![这里写图片描述](http://img.blog.csdn.net/20180311152733881?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**结构**

&emsp;&emsp;单链表的模板类的代码：

```c++
template<class DataType>
class LinkList
{
public:
	LinkList();                     
	LinkList(DataType a[], int n);  
	~LinkList();                    
	int Length();                   
	DataType Get(int i);            
	int Locate(DataType x);         
	void Insert(int i, DataType x); 
	DataType Delete(int i);         
	void PrintList();               
private:
	Node<DataType> *first;          
};
```

**特点**：

- 用一组任意的存储单元存储线性表的数据元素， 这组存储单元可以存在内存中未被占用的任意位置
- 顺序存储结构每个数据元素只需要存储一个位置就可以了，而链式存储结构中，除了要存储数据信息外，还要存储它的后继元素的存储地址

**无参数构造**

&emsp;&emsp;生成只有头结点的空链表

```c++
template<class DataType>
LinkList<DataType>::LinkList()
{
	first = new Node<DataType>;
	first->next = NULL;
}
```

**头插法构造单链表**

&emsp;&emsp;头插法是每次将新申请的结点插在头结点后面

![这里写图片描述](http://img.blog.csdn.net/20180311152822301?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template<class DataType>
LinkList<DataType>::LinkList(DataType a[], int n)
{
	first = new Node<DataType>;
	first->next = NULL;
	for (int i = 0; i < n; i++)
	{
		Node<DataType> *s = new Node<DataType>;
		s->data = a[i];
		s->next = first->next;
		first->next = s;
	}
}
```

**尾插法构造单链表**

&emsp;&emsp;尾插法就是每次将新申请的结点插在终端节点的后面

![这里写图片描述](http://img.blog.csdn.net/20180311152858388?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template<class DataType>
LinkList<DataType>::LinkList(DataType a[], int n)
{
	first = new Node<DataType>;
	Node<DataType> *r = first;
	for (int i = 0; i < n; i++)
	{
		Node<DataType> *s = new Node<DataType>;
		s->data = a[i];
		r->next = s;
		r = s;
	}
    r->next = NULL;
}
```

**析构函数**

&emsp;&emsp;单链表类中的结点是用new申请的，在释放的时候无法自动释放，所以，析构函数要将单链表中的结点空间释放

```c++
template<class DataType>
LinkList<DataType>::~LinkList()
{
	while (first != NULL)
	{
		Node<DataType>* q = first;
		first = first->next;
		delete q;
	}
}
```

**计算长度**

&emsp;&emsp;单链表中不能直接求出长度，所以我们只能将单链表扫描一遍，所以时间复杂度为$$O(n)$$ 

![这里写图片描述](http://img.blog.csdn.net/20180311152953343?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template<class DataType>
int LinkList<DataType>::Length()
{
	Node<DataType>* p = first->next;
	int count = 0;
	while (p != NULL)
	{
		p = p->next;
		count++;
	}
	return count;
}
```

**按位查找**

&emsp;&emsp;单链表中即使知道节点位置也不能直接访问，需要从头指针开始逐个节点向下搜索，平均时间性能为$$O(n)$$ ，单链表是**顺序存取**结构

```c++
template<class DataType>
DataType LinkList<DataType>::Get(int i)
{
	Node<DataType>* p = first->next;
	int count = 1;
	while (p != NULL && count<i)
	{
		p = p->next;
		count++;
	}
	if (p == NULL) throw "Location";
	else return p->data;
}
```

**按值查找**

&emsp;&emsp;单链表中按值查找与顺序表中的实现方法类似，对链表中的元素依次进行比较，平均时间性能为$$O(n)$$ .

```c++
template<class DataType>
int LinkList<DataType>::Locate(DataType x)
{
	Node<DataType> *p = first->next;
	int count = 1;
	while (p != NULL)
	{
		if (p->data == x) return count;
		p = p->next;
		count++;
	}
	return 0;
}
```

**插入**

&emsp;&emsp;单链表在插入过程中需要注意分析在表头、表中间、表尾的三种情况，由于单链表带头结点，这三种情况的操作语句一致，不用特殊处理，时间复杂度为$$O(n)$$ 

![这里写图片描述](http://img.blog.csdn.net/20180311153151640?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template<class DataType>
void LinkList<DataType>::Insert(int i, DataType x)
{
	Node<DataType> *p = first;
	int count = 0;
	while (p != NULL && count<i - 1)
	{
		p = p->next;
		count++;
	}
	if (p == NULL) throw "Location";
	else {
		Node<DataType> *s = new Node<DataType>;
		s->data = x;
		s->next = p->next;
		p->next = s;
	}
}
```

**删除**

&emsp;&emsp;删除操作时需要注意表尾的特殊情况，此时虽然被删结点不存在，但其前驱结点却存在。因此仅当被删结点的前驱结点存在且不是终端节点时，才能确定被删节点存在，时间复杂度为$$O(n)$$ .

![这里写图片描述](http://img.blog.csdn.net/20180311153330100?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template<class DataType>
DataType LinkList<DataType>::Delete(int i)
{
	Node<DataType> *p = first;
	int count = 0;
	while (p != NULL && count<i - 1)
	{
		p = p->next;
		count++;
	}
	if (p == NULL || p->next == NULL) throw "Location";
	else {
		Node<DataType> *q = p->next;
		int x = q->data;
		p->next = q->next;
		return x;
	}
}
```

**遍历**

&emsp;&emsp;遍历单链表时间复杂度为$$O(n)$$ .

![这里写图片描述](http://img.blog.csdn.net/20180311153433777?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```c++
template<class DataType>
void LinkList<DataType>::PrintList()
{
	Node<DataType> *p = first->next;
	while (p != NULL)
	{
		cout << p->data << endl;
		p = p->next;
	}
}
```

&emsp;&emsp;完整代码示例(更多数据结构完整示例可见[GitHub](https://github.com/Quanfita/Data_Stuctures))：

```C++
#include<iostream>
using namespace std;

template<class DataType>
struct Node
{
	DataType data;
	Node<DataType> *next;
};

template<class DataType>
class LinkList
{
public:
	LinkList();                     
	LinkList(DataType a[], int n);  
	~LinkList();                    
	int Length();                   
	DataType Get(int i);            
	int Locate(DataType x);         
	void Insert(int i, DataType x); 
	DataType Delete(int i);         
	void PrintList();               
private:
	Node<DataType> *first;          
};

template<class DataType>
LinkList<DataType>::LinkList()
{
	first = new Node<DataType>;
	first->next = NULL;
}

template<class DataType>
LinkList<DataType>::LinkList(DataType a[], int n)
{
	first = new Node<DataType>;
	first->next = NULL;
	for (int i = 0; i < n; i++)
	{
		Node<DataType> *s = new Node<DataType>;
		s->data = a[i];
		s->next = first->next;
		first->next = s;
	}
}

template<class DataType>
LinkList<DataType>::~LinkList()
{
	while (first != NULL)
	{
		Node<DataType>* q = first;
		first = first->next;
		delete q;
	}
}

template<class DataType>
int LinkList<DataType>::Length()
{
	Node<DataType>* p = first->next;
	int count = 0;
	while (p != NULL)
	{
		p = p->next;
		count++;
	}
	return count;
}

template<class DataType>
DataType LinkList<DataType>::Get(int i)
{
	Node<DataType>* p = first->next;
	int count = 1;
	while (p != NULL && count<i)
	{
		p = p->next;
		count++;
	}
	if (p == NULL) throw "Location";
	else return p->data;
}

template<class DataType>
int LinkList<DataType>::Locate(DataType x)
{
	Node<DataType> *p = first->next;
	int count = 1;
	while (p != NULL)
	{
		if (p->data == x) return count;
		p = p->next;
		count++;
	}
	return 0;
}

template<class DataType>
void LinkList<DataType>::Insert(int i, DataType x)
{
	Node<DataType> *p = first;
	int count = 0;
	while (p != NULL && count<i - 1)
	{
		p = p->next;
		count++;
	}
	if (p == NULL) throw "Location";
	else {
		Node<DataType> *s = new Node<DataType>;
		s->data = x;
		s->next = p->next;
		p->next = s;
	}
}

template<class DataType>
DataType LinkList<DataType>::Delete(int i)
{
	Node<DataType> *p = first;
	int count = 0;
	while (p != NULL && count<i - 1)
	{
		p = p->next;
		count++;
	}
	if (p == NULL || p->next == NULL) throw "Location";
	else {
		Node<DataType> *q = p->next;
		int x = q->data;
		p->next = q->next;
		return x;
	}
}

template<class DataType>
void LinkList<DataType>::PrintList()
{
	Node<DataType> *p = first->next;
	while (p != NULL)
	{
		cout << p->data << endl;
		p = p->next;
	}
}

int main()
{
	LinkList<int> p;
	p.Insert(1, 6);
	p.Insert(2, 9);
	p.PrintList();
	p.Insert(2, 3);
	p.PrintList();
	cout << p.Get(2) << endl;
	cout << p.Locate(9) << endl;
	cout << p.Length() << endl;
	p.Delete(1);
	p.PrintList();
	return 0;
}
```



#### 链式存储的优缺点

**优点**：

1. 插入、删除不需移动其他元素，只需改变指针.
2. 链表各个节点在内存中空间不要求连续，空间利用率高 

**缺点**：

1. 查找需要遍历操作，比较麻烦





## 五、其他线性表

### 循环链表

&emsp;&emsp;循环链表是另一种形式的链式存储结构。它的特点是表中最后一个结点的指针域指向头结点，整个链表形成一个环。（通常为了使空表和非空表的处理一致，通常也附加一个头结点）

![这里写图片描述](http://img.blog.csdn.net/20180311153541327?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

&emsp;&emsp;在很多实际问题中，一般都使用**尾指针**来指示循环链表，因为使用尾指针查找开始结点和终端结点都很方便。

![这里写图片描述](http://img.blog.csdn.net/20180311153630144?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

&emsp;&emsp;循环链表没有增加任何存储量，仅对链接方式稍作改变，循环链表仅在循环条件与单链表不同。从循环链表的任一结点出发可扫描到其他结点，增加了灵活性。但是，由于循环链表没有明显的尾端，所以链表操作有进入死循环的危险。通常以判断指针是否等于某一指定指针来判定是否扫描了整个循环链表。



### 双链表

&emsp;&emsp;循环链表虽然可以从任意结点出发扫描其他结点，但是如果要查找其前驱结点，则需遍历整个循环链表。为了快速确定任意结点的前驱结点，可以再每个节点中再设置一个指向前驱结点的指针域，这样就形成了**双链表**。

![这里写图片描述](http://img.blog.csdn.net/20180311153714830?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**存储方法**

```c++
template<class DataType>
struct Node
{
	DataType data;
	Node<DataType> *prior,*next;
};
```

&emsp;&emsp;结点p的地址既存储在其前驱结点的后继指针域内，又存储在它后继结点的前驱指针域中

需要注意：

1. 循环双链表中求表长、按位查找、按值查找、遍历等操作的实现与单链表基本相同。
2. 插入操作需要修改4个指针，并且要注意修改的相对顺序。



### 静态链表

&emsp;&emsp;静态链表是用数组来表示单链表，用数组元素的下标来模拟单链表的指针。

**静态链表的存储结构**：

```c++
const int MaxSize = 100;
template<class DataType>
struct Node{
	DataType data;
    int next;
}SList[MaxSize];
```

**静态链表存储示意图**：

![这里写图片描述](http://img.blog.csdn.net/20180311153823776?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**静态链表插入操作示意图**：

![这里写图片描述](http://img.blog.csdn.net/20180311153857775?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**静态链表删除操作示意图**：

![这里写图片描述](http://img.blog.csdn.net/20180311153938316?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

&emsp;&emsp;静态链表虽然是用数组来存储线性表的元素，但在插入和删除操作时，只需要修改游标，不需要移动表中的元素，从而改进了在顺序表中插入和删除操作需要移动大量元素的缺点，但是它并没有解决连续存储分配带来的表长难以确定的问题。



### 间接寻址

&emsp;&emsp;间接寻址是将数组和指针结合起来的一种方法，它将数组中存储的单元改为存储指向该元素的指针。

![这里写图片描述](http://img.blog.csdn.net/20180311154017416?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

&emsp;&emsp;该算法的时间复杂度仍为$$O(n)$$ ，但当每个元素占用较大空间时，比顺序表的插入快的多。线性表的间接寻址保持了顺序表随机存取的优点，同时改进了插入和删除操作的时间性能，但是它也没有解决连续存储分配带来的表长难以确定的问题。



&emsp;&emsp;具体代码实现均可在[GitHub](https://github.com/Quanfita/Data_Stuctures)中找到。如有错误，请在评论区指正。



## 参考

- 数据结构（C++版）王红梅等编著