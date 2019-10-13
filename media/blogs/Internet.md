---
title: 计算机网络笔记整理——绪论
categories:
- note
tags: 
- 计算机网络
copyright: true
mathjax: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 因特网

所有接入因特网的设备都称为**主机 **或**端系统**。

主机通过**通信链路**和**分组交换机**连接，**路由器**和**链路层交换机**都是分组交换机。

一个分组所经历的一系列通信链路和分组交换机称为通过该网络的**路径**。

**ISP**是因特网服务供应商，有不同层级，每个ISP是一个由多个分组交换机和多段通信链路组成的网络。

**协议**控制着因特网中信息的接收和发送。因特网的主要协议统称为**TCP/IP**。

- TCP 可靠 面向连接  确保传递与流量控制
- IP UDP 不可靠 非面向连接

![这里写图片描述](http://img.blog.csdn.net/20180311155659592?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**因特网的两种描述方法**：一种是根据它的硬件和软件组成来描述，另一种是根据基础设施向分布式应用程序提供的服务来描述。

**通信链路**有不同的物理媒体组成，不同媒体具有不同速率，包括同轴电缆、铜线、光缆、无线电

**分组**： 当一台端系统有数据要向另一台端系统发送时，端系统将数据分段并在每段加上首部字节，由此形成的信息包称为**分组**。

一个**协议**定义了在两个或多个通信实体之间交换的报文格式和次序，以及在报文或其他事件方面所采取的动作、传输和/或接收。

**端系统 = 主机**：和因特网相连的计算机等设备（如TV，Web服务器，手提电脑）。

主机有时候有进一步分为两类：**客户机（client）**和**服务器（server）**。

在网络软件的上下文中，客户机和服务器有另一种定义，**客户机程序（client program）**是运行在一个端系统上的程序，它发出请求，并从运行在另一个端系统上的**服务器程序（server program）**接收服务。

客户机-服务器因特网应用程序是**分布式应用程序（distributed application）**。

还有的应用程序是P2P对等应用程序，其中的端系统互相作用并运行执行客户机和服务器功能的程序。



##网络边缘

互联网的边缘部分也叫**资源子网**。

**接入网**：将端系统连接到**边缘路由器（edge router）**的物理链路。

**边缘路由器**：端系统到任何其他远程端系统的路径上的第一台路由器。

![这里写图片描述](http://img.blog.csdn.net/20180311155759387?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 网络接入类型

网络接入的三种类型：（分类并不严格）

1. 住宅接入（residential  access），将**家庭端系统（PC或家庭网络）**和网络相连。

   ​    1. 通过普通模拟电话线用**拨号调制解调器（dial-up modem）**与住宅ISP相连。家用调制解调器将PC输出的数字信号转化为模拟形式，以便在模拟电话线上传输。模拟电话线由**双绞铜线**构成，就是用于打普通电话的电话线。允许56kbps接入（下载上传速度慢），用户上网就不能打电话了。

   ​    2. **数字用户线（digital subscriber line，DSL）**:一种新型的调制解调器技术，类似于**拨号调制解调器**，也运行在现有的双绞线电话线上，通过限制用户和ISP调制解调器之间的距离，DSL能够以高得多的速率传输和接受数据。（使用频分多路复用技术），分为上行信道和下行信道，两个信道速率不一样。

   ​    3. **混合光纤同轴电缆（hybrid fiber-coaxial cable， HFC）**：使用了光缆和同轴电缆相结合的技术。扩展了当前用于广播电缆电视的电缆网络，需要**电缆调制解调器（cable modem），**分为上行信道和下行信道，共享广播媒体（HFC特有），信道都是共享的，需要一个分布式多路访问协议，以协调传输和避免碰撞。

2. 公司接入（company access），将**商业或教育机构中的端系统**和网络相连

   ​    1. 局域网（LAN）

   ​    2. 以太网

   ​        1. 共享以太网

   ​        2. 交换以太网

3. 无线接入（wireless access），将**移动端系统**与网络相连。分为两类

   ​    1. **无线局域网（wireless LAN）：**无线用户与位于几十米半径内的基站（无线接入点）之间传输/接收分组。这些基站和有线的因特网相连接，因而为无线用户提供连接到有线网络的服务。

   2. **广域无线接入网（wide-area wireless access network）**：分组经用于蜂窝电话的相同无线基础设施进行发送，基站由电信提供商管理，为数万米半径内的用户提供无线接入服务。



基于IEEE 802.11技术的无线局域网也被称为无线以太网和WiFi。

HFC使用了光缆和同轴电缆相结合的技术，拨号56kbps调制解调器和ASDL使用了双绞铜线；移动接入网络使用了无线电频谱。



### 物理媒体

物理媒体分为两类：

​    1. 导引型媒体（guided media）：电波沿着**固体媒体**（光缆，双绞铜线或同轴电缆）被导引。

2. 非导引型媒体（unguided media）：电波在**空气或外层空间**（在无线局域网或数字卫星频道）中传播；





导引型媒体（guided media）

​    1. **双绞铜线**：最便宜，使用最普遍，两根线被绞合起来，以减少对邻近双绞线的电气干扰。一根电缆由许多双绞线捆扎在一起，并在外面覆盖上保护性防护层，一堆电线构成一个通信链路。**非屏蔽双绞线UTP**常用于建筑物内的计算机网络中，即用于局域网（LAN）中。双绞线最终已经成为高速LAN联网的主要方式。

​   2. **同轴电缆**：能作为导引式共享媒体，具有高比特速率。

3. **光缆**：不受电磁干扰，长达100km的光缆信号衰减极低，并且很难接头。





非导引型媒体（unguided media）：电波在**空气或外层空间**（在无线局域网或数字卫星频道）中传播；

​    1. **陆地无线电信道**：具有穿透墙壁，提供与移动用户的连接以及长距离承载信号的能力。

2. **卫星无线电信道**


   一颗通信卫星连接两个或多个位于地球的微波发射方/接收方，它们被称为地面站。卫星无线电信道分为**同步卫星**和**低地球轨道卫星**。

   同步卫星永久的停留在地球上方相同的点，卫星链路常用于电话网或因特网的主干。卫星链路在那些无法使用DSL或基于电缆的因特网接入区域，也越来越多地用作高速住宅因特网接入。

   低地球轨道卫星 围绕地球旋转，彼此通信，未来低地球轨道卫星技术也许能用于因特网接入。



## 网络核心

网络核心，即互联了**因特网端系统的分组交换机**和**链路的网状网络**，也叫**通信子网**。

![这里写图片描述](http://img.blog.csdn.net/20180311155906494?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

通过网络链路和交换机移动数据有两种基本方法：**电路交换（circuit switching）**和**分组交换（packet swiitching）**。

**电路交换网络**中，沿着端系统通信路径，为端系统之间通信所提供的资源在通信会话期间会被预留。例子有电话网络。

**分组交换网络**中，这些资源不被预留。例子有因特网网络。

1. 电路交换：创建专用的端到端连接，独占带宽 有独立的建立、连接过程；

2. 电路交换网络中的多路复用

   ​    1. 频分多路复用（Frequency-Division Multiplexing，FDM）

   ​    2. 时分多路复用（Time-Division Multiplexing，TDM）

3. 分组交换：共享带宽

   ​    存储转发传输机制：在交换机能够开始向输出链路传输该分组的第一个比特之前，必须接收到整个分组。

4. 分组交换和电路交换对比：统计多路复用



### 分组交换

​	分组交换是以分组为单位进行传输和交换的，它是一种存储——转发交换方式，即将到达交换机的分组先送到存储器暂时存储和处理，等到相应的输出电路有空闲时再送出。

![这里写图片描述](http://img.blog.csdn.net/20180311155922915?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

​	**优点**：

​	（1）分组交换不需要为通信双反预先建立一条专用的通信线路，不存在连接建立时延，用户可随时发送分组。

​	（2）由于采用存储转发方式，加之交换节点具有路径选择，当某条传输线路故障时可选择其他传输线路，提高了传输的可靠性。

​	（3）通信双反不是固定的战友一条通信线路，而是在不同的时间一段一段地部分占有这条物理通路，因而大大提高了通信线路的利用率。

​	（4）加速了数据在网络中的传输。因而分组是逐个传输，可以使后一个分组的存储操作与前一个分组的转发操作并行，这种流水线式传输方式减少了传输时间。

​	（5）分组长度固定，相应的缓冲区的大小也固定，所以简化了交换节点中存储器的管理。

​	（6）分组较短，出错几率减少，每次重发的数据量也减少，不仅提高了可靠性，也减少了时延。

![这里写图片描述](http://img.blog.csdn.net/20180311160209121?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

​	**缺点：**

​	（1）由于数据进入交换节点后要经历存储转发这一过程，从而引起的转发时延（包括接受分组、检验正确性、排队、发送时间等），而且网络的通信量越大，造成的时延就越大，实时性较差。

​	（2）分组交换只适用于数字信号。

​	（3）分组交换可能出现失序，丢失或重复分组，分组到达目的节点时，对分组按编号进行排序等工作，增加了麻烦。

​	**综上**，若传输的数据量很大，而且传送时间远大于呼叫时间，则采用电路交换较为合适；当端到端的通路有很多段链路组成是，采用分组交换较为合适。从提高整个网络的信道利用率上看，分组交换优于电路交换。



### 电路交换

​	电路交换是以电路连接为目的的交换方式，通信之前要在通信双方之间建立一条被双方独占的物理通道。

​	**电路交换的三个阶段**：

​	（1）建立连接	（2）通信	（3）释放连接

​	**优点**：

​	（1）由于通信线路为通信双方用户专用，数据直达，所以传输数据的时延非常小。

​	（2）通信双方之间的屋里通路一旦建立，双方可以随时通信，实时性强。

​	（3）双方通信时按发送顺序传送数据，不存在失序问题。

​	（4）电路交换既适用于传输模拟信号，也适用于传输数字信号。

​	（5）电路交换的交换设备及控制均比较简单。

​	**缺点**：

​	（1）电路交换平均连接建立时间对计算机通信来说较长。

​	（2）电路交换家里连接后，物理通路被通信双方独占，即使通信线路空闲，也不能供其他用户使用，因而信道利用率低。

​	（3）电路交换时，数据直达，不同类型，不同规格，不同速率的终端很难相互进行通信，也难以在通信过程中进行差错控制。



## 时延、丢包和吞吐量

时延分为**节点处理时延（nodal processing delay），排队时延（queuing delay），传输时延（transmission delay）**和**传播时延（propagation delay）**，这些加起来就是**节点总时延（total nodal delay），即**

**节点总时延 = 节点处理时延 + 排队时延 + 传输时延 + 传播时延**



### 时延类型

1. **处理时延**

   1. 检查分组首部和决定将分组导向哪一个队列；
   2. 其他：检查比特级差错所需要的时间。

2. **排队时延**

   在队列中，当分组在链路上等待传输时所需的时间，取决于先期到达的，正在排队等待想链路传输分组的数量。

3. **传输时延**

   1. 将所有分组的比特推向链路所需要的时间。
   2. 用L比特表示分组的长度，用R bps表示从路由器A到路由器B的链路传输速率。（对于一条10Mbps的以太网链路，速率R = 10Mbps）,**传输时延（**又称为**存储转发时延）**是**L/R**。

4. **传播时延**

   ​    1. 从该链路的起点到路由器B传播所需要的时间是**传播时延**。该比特以该链路的传播速率传播。

   ​    2. 传播时延 = 两台路由器的距离d / 传播速率s。

   3. 传播速率取决于该链路的物理媒体（即光纤，双绞铜线等），速率范围是$$2\times10^8\sim3\times10^8 m/s$$



### 端到端时延

假定在源主机和目的主机之间有$$N-1$$台路由器，并且该网络是无拥塞的（因此排队时延是微不足道的），处理时延为$$d_{proc}$$ ，每台路由器和源主机的输出速率是 R bps，每条链路的传播时延是$$d_{proc}$$ ，节点时延累加起来得到端到端时延

$$d_{end-end} = N（d_{proc} + d_{trans} + d_{prop}）$$ 

$$d_{trans} = L /R$$ 

1. **Traceroute程序**

   能够在任何因特网主机上运行。当用户指定一个目的主机名字时，元主机中的改程序朝着该目的地发送多个特殊分组之一时，它向源回送一个短报文，该报文包括该路由器的名字和地址。RFC1393描述了Traceroute。

2. **端系统、应用程序和其他时延**

​    除了处理时延，传输时延，传播时延外，端系统中还有一些其他重要的时延：

​    1. 拨号调制解调器引入的**调制/编码时延**，量级在几十毫秒，对于以太网，电缆调制解调器和DSL等接入技术，这种时延是不太多的；

​    2. 向共享媒体传输分组的端系统可以将有意地延迟传输作为其协议的一部分，以便与其他端系统共享媒体。（第五章探讨）

​    3. **媒体分组化时延**，在IP话音（VoIP）应用中。在VoIP中，发送方在向因特网传递分组之前必须首先用编码的数字化语音填充分组，这种填充分组的时间就是**分组化时延**。（可能比较大）



### 计算机网络中的吞吐量

​    除了时延和丢包外，计网中另个一个必不可少的性能测度就是**端到端吞吐量**。

​    吞吐量分为**瞬时吞吐量（instancous throughput）**和**平均吞吐量（average throughput）**，我们可以把他们类比为以前物理学过的瞬时速度和平均速度。

​    瞬时吞吐量是主机B接受到该文件的一个速率，平均吞吐量是所有比特F/T秒，即F/T bps。

​    对于某些应用程序（譬如因特网电话），希望他们具有低时延，并保持高于某一阈值的一致的瞬时吞吐量，对于其他应用程序（譬如文件传输等等），时延不是很重要，但是希望能具有尽可能高的吞吐量。

​    吞吐量：单位时间内通过某个网络（或信道、接口）的数据量。



## 协议层次

因特网协议栈：

应用层、运输层、网络层、数据链路层、物理层。

OIS的七层协议：

应用层、表示层、会话层、运输层、网络层、数据链路层、物理层。

TCP/IP四层协议：

应用层、运输层、网际层、网络接口层



![这里写图片描述](http://img.blog.csdn.net/20180311160251696?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



**物理层**：最基础的一层，建立在传输媒介基础上，起到**建立、维护和取消物理连接的作用**，实现设备之间的物理接口。物理层只接收和发送一串比特流，不考虑信息的意义和信息结构。包括对连接到网络上的设备描述其各种机械的、电气的、功能的规定。**典型设备有：光纤、同轴电缆、双绞线、中继器和集线器**。

**数据链路层**：在物理层提供比特流服务的基础上，将比特信息封装成数据帧Frame，起到在物理层上建立、撤销、标识逻辑链接和链路复用以及差错校验等功能。通过使用接收系统的硬件地址或物理地址来寻址。建立相邻结点之间的数据链接，通过差错控制提供数据帧（Frame）在信道上无差错的传输，同时为其上面的网络层提供有效的服务。数据链路层在不可靠的物理介质上提供可靠的传输。该层的作用包括：**物理地址寻址、数据的成帧、流量控制、数据检错、重发等**。**典型设备有：二层交换机、网桥、网卡**。**差错控制**：在数据传输过程中如何发现并更在错误；流量控制：通信双方速度存在差异，需要协调匹配通信正常。

**网络层**：或通信子网层，是高层协议之间的界面层，用于控制通信子网的操作，是通信子网与资源子网的接口。在网络间进行通信的计算机之间可能会通过多个数据链路，也可能还要经过很多通信子网。网络层的任务就是选择合适的网间路由和交换结点，确保数据及时传送。网络层将解封装数据链路层收到的帧，提取数据包，包中封装有网络层包头，其中含有逻辑地址信息，包括源站点和目的站点地址的网络地址。**典型设备是路由器**。
网络层主要功能为**管理数据通信**，**实现端到端的数据传送服务**；**主体协议是IP协议**

**运输层**：建立在网络层和会话层之间，实质上它是网络体系结构中低层与高层之间衔接的一个接口层。用一个寻址机制来标识一个特殊的应用程序（端口号）。传输层不仅是一个独立的结构层，它还是整个分层体系结构的核心。传输层的数据单元是由数据组织成的数据段（Segment）这个层负责获取全部信息，因此它必须跟踪数据单元碎片、乱序到达的数据包和其他在传输过程中可能发生的危险。
主要功能为负责总体的**数据传输**和**数据控制**，**主要包括两个协议：TCP：传输控制协议；UDP：用户报文协议**

**会话层**：也称会晤层或对话层，在会话层及以上的高层次中，数据传达的单位不再另外命名，统称为报文。会话层不参与具体传送，它提供包括访问校验和会话管理在内的建立和维护应用之间通信的机制。会话层提供的服务可使应用建立和维持会话，并使会话同步，会话层使用校验点可以使通信会话在通信失效时从校验点继续恢复通信，这对传送大型文件极为重要。
主要功能是**为通信进程建立连接**。

**表示层**：对上服务应用层，对下接收会话层的服务，是为应用过程之中传送的信息提供表示方法的服务，它关心的只是发出的信息的语义和语法。表示层要完成某些特定的功能，主要有不同的数据编码格式的转换，提供数据压缩、解压缩服务，对数据进行加密、解密。如图像格式的显示就是由位于表示层的协议来支持的。表示层提供的服务包括：**语法选择、语法转换**等，语法选择是提供一种初始语法和以后修改这种选择的手段。语法转换涉及代码转换和字符集的转换、数据格式的修改以及对数据结构操作的适配。主要功能是进行**加密和压缩**。

**应用层**：是通信用户之间的窗口，为用户提供网络管理、文件传输、事务处理等服务。其中包含了若干独立的用户通用的服务协议模块。网络应用层是OSI的最高层，**为网络用户之间的通信提供专用的程序**。主要功能是**为通信进程建立连接**。



## 攻击威胁下的网络

1. 坏家伙能够经因特网将恶意软件放入你的计算机
2. 坏家伙能够攻击服务器和网络基础设施
3. 能够嗅探分组
4. 能够伪装成信任的人：IP哄骗（IP spoofing）可用端点鉴别（end-point authentication）机制
5. 修改或删除报文：中间人攻击



恶意软件（malware）：删除文件、收集隐私信息

僵尸网络（botnet）：对目标主机展开垃圾邮件分发或分布式拒绝服务攻击。

病毒（virus）：恶意可执行代码并自我复制

蠕虫（worm）：不需明显交互即可感染

特洛伊木马（Trojan horse）：为进行非法目的的计算机病毒



## 计算机网络发展史

1. 分组交换 1961-1972

   第一个分组交换网ARPANET

2. 专用网络和网络互联 1972-1980

   开始产生TCP、UDP等协议

3. 网络的激增 1980-1990

   建立CSFNET

4. 因特网爆炸 20世纪90年代 
   万维网应用出现



## 参考文章

[计算机网络－笔记](https://www.jianshu.com/p/21f3af4653bb)

[电路交换与分组交换的区别](http://blog.csdn.net/liuqiyao_01/article/details/39001067)

[《计算机网络自顶向下方法》读书笔记](http://www.cnblogs.com/ArtemisZ/p/7555079.html)
