---
title: GPSR协议的NS2仿真全过程（环境+实验）
categories: 
- note
tags:
- GPSR
- NS2
copyright: true
---

前些日子帮老师做了个NS2仿真的小项目，现在项目做完了，写篇博客把流程记录下来。做项目时，NS2和GPSR相关的东西找了好久，总会遇到问题，希望我这篇博客能给广大同学们带来点帮助吧。

----------


##NS2环境搭建

###软硬件环境概述

- Windows10(x64)
- VMware Workstation Pro 12.5
- LinuxMint 18.1
- ns-allinone-2.35

###环境搭建过程

首先，在官网下载ns-allinone-2.35.tar.gz压缩包(<http://sourceforge.net/projects/nsnam/files/allinone/ns-allinone-2.35/ns-allinone-2.35.tar.gz/download>)，再下载GPSR源码，我选择的是CSDN上的KeLiu版(<http://download.csdn.net/download/joanna_yan/8474651>)。

####NS2安装

按Ctrl+Alt+T，打开终端

依次输入

>  sudo apt-get update
>
>   sudo apt-get upgrade
>
>   sudo apt-get install build-essential
>
>   sudo apt-get install tcl8.5 tcl8.5-dev tk8.5 tk8.5-dev 
>
>   sudo apt-get install libxmu-dev libxmu-headers
>
>  tar xvfz ns-allinone-2.35.tar.gz
>
>  cd ns-allinone-2.35
>
>   sudo ./install

在安装的时候会报个错，这是由于源码gcc版本比较老
![这里写图片描述](http://img.blog.csdn.net/20180124093026094?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
修改只要在linkstate/ls.h文件137行
![这里写图片描述](http://img.blog.csdn.net/20180124093059525?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
修改成
![这里写图片描述](http://img.blog.csdn.net/20180124093136166?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
 然后重新安装
![这里写图片描述](http://img.blog.csdn.net/20180124093158307?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
如上图所示，这样就安装成功了。

然后我们需要配置环境变量，否则无法启动。

>  sudo gedit /home/(用户名)/.bashrc

在最后加上下面语句，用户名换成自己的即可

```bash
export PATH="$PATH:/home/(用户名)/ns-allinone-2.35/bin:/home/(用户名)/ns-allinone-2.35/tcl8.5.10/unix:/home/(用户名)/ns-allinone-2.35/tk8.5.10/unix"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/(用户名)/ns-allinone-2.35/otcl-1.14:/home/(用户名)/ns-allinone-2.35/lib"

export TCL_LIBRARY="$TCL_LIBRARY:/home/(用户名)/ns-allinone-2.35/tcl8.5.10/library"

```
![这里写图片描述](http://img.blog.csdn.net/20180124093244398?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
 修改完毕，保存，关闭当前终端，再打开一个新的终端，输入ns，回车，如果显示一个%，就证明ns2安装成功了。
![这里写图片描述](http://img.blog.csdn.net/20180124093339264?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

####NAM安装

终端输入nam，如果能够出现nam的窗口则nam可以正常使用，如果提示nam没有安装或者是不能识别的命令，cd /home/ns-allinone-2.35/nam.1.15，ls看看是否有nam文件，如果有的话cp nam ../bin，把nam命令复制到bin中。如果没有的话，sudo ./configure，再sudo make，现在得到了nam，再把nam命令复制到bin中。 接着在终端输入nam检验是否可以运行。

----------


##开始实验

###添加GPSR协议

我们搭建好了NS2仿真平台，现在就可以把我们准备好的协议源码解压，放到ns-2.35目录下，然后对ns2的代码进行修改，使我们的协议可以正常运行。

```c++
1. 进入$HOME/ns-allinone-2.30/ns-2.30/common,修改packet.h

enum packet_t{

      //增加 PT_GPSR  

      }

class p_info {

    //增加 name_[PT_GPSR]= “gpsr”

   }

2. 进入$HOME/ns-allinone-2.30/ns-2.30/trace,修改 cmu-trace.cc

 void CMUTrace::format(Packet* p, const char *why) 
{

     //增加 case PT_GPSR;
             break;
}

3. 进入$HOME/ns-allinone-2.30/ns-2.30/queue，修改priqueue.cc

void  PriQueue::recv(Packet *p, Handler *h) 
{

   //增加  case PT_GPSR:

}

```

```tcl
4. 进入$HOME/ns-allinone-2.30/ns-2.30/tcl/lib,修改ns-packet.tcl

foreach prot{

   #增加GPSR

 }
```

```makefile
5. 进入$HOME/ns-allinone-2.30/ns-2.30/ ，修改Makefile

OBJ_STL = #最后按照格式加入（ gpsr前为TAB键而不是空格）

 gpsr/gpsr_neighbor.o\

 gpsr/gpsr_sinklist.o\

 gpsr/gpsr.o

#如果需要加入调试信息，则在CCOPT =  -Wall 加上 -g,  如下：

 CCOPT =  -g  -Wall

```

>  6.重新编译，执行如下命令
>
>    cd $HOME/ns-allinone-2.30/ns-2.30/common
>
>    touch packet.cc
>
>    cd ..
>
>    sudo make clean
>
>    sudo make

### 修改协议源码

在仿真过程中，发现KeLiu版GPSR协议在移动场景下存在一些问题，我们进行如下修改：

```c++
1、gpsr.h文件：

90行左右：

class GPSRUpdateSinkLocTimer : publicTimerHandler {

public:

 GPSRUpdateSinkLocTimer(GPSRAgent *a) : TimerHandler() {a_=a;} <---这7行

protected:

 virtual void expire(Event *e);

 GPSRAgent *a_;

};

 

class GPSRQueryTimer : public TimerHandler{

public:

 GPSRQueryTimer(GPSRAgent *a) : TimerHandler() {a_=a;}

protected:

 virtual void expire(Event *e);

 GPSRAgent *a_;

};

 

106行左右：

 

 friend class GPSRHelloTimer;

 friend class GPSRQueryTimer;

 friend class GPSRUpdateSinkLocTimer;
  

 MobileNode *node_;            //the attached mobile node

 PortClassifier *port_dmux_;   //for the higher layer app de-multiplexing
 

125行左右：

GPSRHelloTimer hello_timer_;

 GPSRQueryTimer query_timer_;

 GPSRUpdateSinkLocTimer update_sink_loc_timer_; <---这行

 

  intplanar_type_; //1=GG planarize, 0=RNG planarize

 

 double hello_period_;

 double query_period_;

 double start_update_time_; <---

 double update_sink_loc_period_; <---这2行

 

 void turnon();              //setto be alive

 void turnoff();             //setto be dead

 void startSink();          

  voidstartSink(double);

165行左右：

void hellotout();                //called bytimer::expire(Event*)

 void querytout();

 void updatesinkloctout(); <---这行

public:

 GPSRAgent();

2、gpsr.cc文件：

70行左右：

void

GPSRQueryTimer::expire(Event *e){

 a_->querytout();

}

void

GPSRUpdateSinkLocTimer::expire(Event *e){

 a_->updatesinkloctout();

}


void

GPSRAgent::hellotout(){

 getLoc();

 nblist->myinfo(my_id,my_x,my_y);

//sink_list->update_sink_loc(my_id,my_x,my_y);

//printf("%f\n",node_->speed());

 hellomsg();

 hello_timer.resched(hello_period);

}


void 

GPSRAgent::updatesinkloctout(){

 getLoc();

 sink_list->update_sink_loc(my_id,my_x,my_y);

 //printf("__\n");

 update_sink_loc_timer.resched(update_sink_loc_period);

}


void

GPSRAgent::startSink(){

 if(sink_list->new_sink(my_id, my_x, my_y, 

 my_id, 0, query_counter))

   querytout();

}


119行左右：


GPSRAgent::GPSRAgent() : Agent(PT_GPSR), 

   hello_timer(this), query_timer(this),

update_sink_loc_timer_(this),

   my_id(-1), my_x(0.0), my_y_(0.0),

   recv_counter(0), query_counter(0),

   query_period_(INFINITE_DELAY)

{

 bind("planar_type", &planar_type);  

 bind("hello_period", &hello_period);

 bind("update_sink_loc_period", &update_sink_loc_period);

 sink_list_ = new Sinks();

 nblist_ = new GPSRNeighbors();


 for(int i=0; i<5; i++)

   randSend_.reset_next_substream();
}

void

GPSRAgent::turnon(){

 getLoc();

 nblist->myinfo(my_id, my_x, my_y);

 hello_timer.resched(randSend.uniform(0.0, 0.5));

 update_sink_loc_timer.resched(start_update_time);

}

void

GPSRAgent::turnoff(){

 hello_timer_.resched(INFINITE_DELAY);

 query_timer_.resched(INFINITE_DELAY);

 update_sink_loc_timer_.resched(INFINITE_DELAY);

}

3、gpsr_sinklist.h文件中：

55行：

class Sinks {

 struct sink_entry *sinklist_;

 public:

 Sinks();

 bool new_sink(nsaddr_t, double, double, nsaddr_t, int, int);

 bool update_sink_loc(nsaddr_t, double, double);

 bool remove_sink(nsaddr_t);

 void getLocbyID(nsaddr_t, double&, double&, int&);

 void dump();

};

4、gpsr_sinklist.cc中：

74行：

 temp->next_ = sinklist_;

 sinklist_ = temp;

 return true;

}

bool

Sinks::update_sink_loc(nsaddr_t id,doublex,double y)

{

 struct sink_entry *temp = sinklist_;

 while(temp){

   if(temp->id_ == id){

 temp->x_ = x;

 temp->y_ = y;

 return true;

}

   temp = temp->next_;

  }

 return false;

}

bool

Sinks::remove_sink(nsaddr_t id){

 struct sink_entry *temp;

 struct sink_entry *p, *q;

```

修改完成后重新编译NS2。 

但是NAM还是有问题，这里是自带的TCL文件的问题，做如下修改就可以了：

```tcl
61行左右：

set opt(tr) trace.tr ;# trace file

set opt(nam)            gpsr.nam <---这里

set opt(rp)             gpsr ;# routing protocol script(dsr or dsdv)

set opt(lm)             "off" ;# log movement


117行左右：（修改比较多，行数不准）

# ======================================================================

# Agent/GPSR setting

Agent/GPSR set planar_type_  1  ;#1=GG planarize, 0=RNG planarize

Agent/GPSR set hello_period_   1.5 ;#Hello message period

Agent/GPSR set update_sink_loc_period_ 0.5

Agent/GPSR set start_update_time_ 0.001

#======================================================================

159行左右：

set tracefd [open $opt(tr) w]

ns_ trace-all  tracefd

 

set namfile [open $opt(nam) w]

ns_ namtrace-all-wireless namfile opt(x)opt(y)


topo load_flatgrid opt(x) $opt(y)

prop topography topo

 

197行左右：


source ./gpsr.tcl


for {set i 0} {i < opt(nn) } {incr i}{

   gpsr-create-mobile-node $i

   ns_ initial_node_pos node_($i) 20

   node_(i) namattach $namfile

}


#

# Source the Connection and Movementscripts

#
 

下面是CBR和场景文件的问题：给个简单的3个节点的例子，在trace文件中可以看到数据包的转发和接收

cbr文件：

# GPSR routing agent settings

for {set i 0} {i < opt(nn)} {incr i} {

   ns_ at 0.00002 "ragent_($i) turnon"

   ns_ at 2.0 "ragent_($i) neighborlist"

#   ns_ at 30.0 "ragent_($i) turnoff"

}

ns_ at 11.2 "ragent_(2) startSink10.0"   #<---这里只要让目标节点startSink就可以，例子是0向2发



set null_(1) [new Agent/Null]

ns_ attach-agent node(2) $null(1)

 

set udp_(1) [new Agent/UDP]

ns_ attach-agent node(0) $udp(1)

 

set cbr_(1) [new Application/Traffic/CBR]

$cbr(1) set packetSize 32

$cbr(1) set interval 2.0

$cbr(1) set random 1

#   $cbr(1) set maxpkts 100

cbr_(1) attach-agent udp_(1)

ns_ connect udp(1) $null(1)

ns_ at 66.0 "cbr_(1) start"

ns_ at 150.0 "cbr_(1) stop"

```

###修改底层协议为802.11p

GPSR是路由协议，也就是工作在网络层的，底层的协议默认应该是IEEE802.11。应该用IEEE802.11p，这个才是针在ns-allinone-2.35/ns-2.35/tcl/ex/802.11目录下找到了IEEE802-11p.tcl文件，里面的设置都是符合IEEE802.11p协议的参数对车载自组网的协议。

所以在wireless-gpsr.tcl中把其他的MAC层和物理层的设置都注释掉，换上IEEE802.11p的设置：

```tcl
94行左右：
#Phy/WirelessPhy set CPThresh_ 10.0

#Phy/WirelessPhy set CSThresh_ 1.559e-11

#Phy/WirelessPhy set RXThresh_ 3.652e-10

#Phy/WirelessPhy set Rb_ 2*1e6

#Phy/WirelessPhy set freq_ 914e+6 

#Phy/WirelessPhy set L_ 1.0

 

# The transimssion radio range 

#Phy/WirelessPhy set Pt_ 6.9872e-4    ;# ?m

#Phy/WirelessPhy set Pt_ 8.5872e-4    ;# 40m

#Phy/WirelessPhy set Pt_ 1.33826e-3   ;# 50m

#Phy/WirelessPhy set Pt_ 7.214e-3     ;# 100m

#Phy/WirelessPhy set Pt_ 0.2818       ;# 250m

 

#802.11p

puts "Loading IEEE802.11pconfiguration..."

source ../tcl/ex/802.11/IEEE802-11p.tcl

puts "Load complete..."

```

###修改无线传输范围

ns-2.35/indep-utils/propagation/下有个threshold工具，可以通过距离、功率等等条件算出这些参数。

我们要先对threshold.cc文件进行修改

\#include "iostream.h"  修改为\#include "iostream"，同时加上using namespace std;

再添加头文件：#include "cstring"

在当前目录终端输入命令，进行编译：

>  cd ns/indep-utils/propagation/
>  g++ -lm threshold.cc -o threshold

编译成功生成threshold文件
![这里写图片描述](http://img.blog.csdn.net/20180124093707890?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
这时我们利用threshold工具就可以得到我们需要的RXThresh的值了。

>  ./threshold -m TwoRayGround -r 1 550

![这里写图片描述](http://img.blog.csdn.net/20180124093744180?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
这就是我们需要的RXThresh数值，将我们新得到的数值替换之前的数值，即可改变我们无线传输的范围了。



参数设置好后，我们就可以开始仿真实验了。（中途我还修改了协议其他地方的代码，涉及到项目内容，在此不过多赘述，所以后面的数据跟GPSR协议本身跑出来的数据不大一样）

###setdest生成随机场景

setdest可以随机产生无线网络仿真所需要的移动场景。

setdest程序放在 ns-2.35/indep-utils/cmu-scen-gen/setdest/目录下

usage：setdest \[-nnodes]\[-p pause]\[-M maxrate]\[-t time]\[-x x]\[-y y]

我们需要50个节点，最大速度10m/s，持续时间20s，场景长250，宽200

>  ./setdest -n 50-p 0.0 -M 10.0 -t 20 -x 250 -y 200 >scen_50_0_10_20_25_20

 

###cbrgen生成数据流

cbrgen工具在ns-2.35/indep-utils/cmu-scen-gen/目录下

usage：cbrgen.tcl \[-type cbr]\[-nn nodes]\[-seed seed]\[-mcconnections]\[-rate rate]

我们此时有50个节点，需要一组数据流所以，输入：

> ns cbrgen.tcl -type cbr -nn 50 -seed 1 -mc 1-rate 1.0 > cbr_n50_m1_r1

 

###重新编写tcl脚本

现在，我们把仿真需要的场景，数据流都准备好了，我们把生成的场景文件移动到gpsr协议的文件夹中，再次修改tcl脚本

添加：

```tcl
source scen_50_0_10_20_25_20

source cbr_n50_m1_r1

```

修改完的脚本是这个样子的(这里我另存为gpsr-wireless.tcl文件)：

```tcl
set opt(chan) Channel/WirelessChannel

set opt(prop) Propagation/TwoRayGround

set opt(netif) Phy/WirelessPhy

set opt(mac) Mac/802_11

set opt(ifq) Queue/DropTail/PriQueue

set opt(ll) LL 

set opt(ant) Antenna/OmniAntenna


set opt(x) 250

set opt(y) 200


set opt(ifqlen) 50

set opt(nn) 50

set opt(seed) 0.0

set opt(stop) 20.0

set opt(tr) trace.tr

set opt(nam) out.nam

set opt(rp) gpsr

set opt(lm) "off"
 

LL set mindelay_          50us

LL set delay_                25us

LL set bandwidth_       0     ;# not used

Agent/Null set sport_          0

Agent/Null set dport_         0

Agent/CBR set sport_          0

Agent/CBR set dport_         0

Agent/UDP set sport_         0

Agent/UDP set dport_         0

Agent/UDP set packetSize_ 1460

Queue/DropTail/PriQueue setPrefer_Routing_Protocols    1

Antenna/OmniAntenna set X_ 200

Antenna/OmniAntenna set Y_ 200

Antenna/OmniAntenna set Z_ 1.5

Antenna/OmniAntenna set Gt_ 1.0

Antenna/OmniAntenna set Gr_ 1.0

#802.11p

puts "Loading IEEE802.11pconfiguration..."

source ../tcl/ex/802.11/IEEE802-11p.tcl

puts "Load complete..."

# Agent/GPSR setting

Agent/GPSR set planar_type_  1  ;#1=GG planarize, 0=RNG planarize

Agent/GPSR set hello_period_   5.0 ;#Hello message period

Agent/GPSR set update_sink_loc_period_ 0.5

Agent/GPSR set start_update_time_ 0.001

source ../tcl/lib/ns-bsnode.tcl

source ../tcl/mobility/com.tcl

set ns_ [new Simulator]

set chan [new $opt(chan)]

set prop [new $opt(prop)]

set topo [new Topography]

 

set tracefd [open $opt(tr) w]

ns_ trace-all tracefd

 

set namfile [open $opt(nam) w]

ns_ namtrace-all-wireless namfile opt(x)opt(y)

topo load_flatgrid opt(x) $opt(y)

prop topography topo

set god_ [create-god $opt(nn)]

$ns_ node-config -adhocRouting gpsr \

                 -llType $opt(ll) \

                 -macType $opt(mac) \

                 -ifqType $opt(ifq) \

                -ifqLen $opt(ifqlen) \

                 -antType $opt(ant) \

                 -propType $opt(prop) \

                 -phyType $opt(netif) \

                 -channelType $opt(chan) \

                 -topoInstance $topo \

                 -agentTrace ON \

                 -routerTrace ON \

                 -macTrace OFF \

                 -movementTrace OFF 

source ./gpsr.tcl

for {set i 0} {i < opt(nn)} {incr i} {

       gpsr-create-mobile-node$i;

       node_(i)namattach $namfile;

}

source scen_50_0_10_20_25_20

source cbr_n50_m1_r1


for {set i 0} {i < opt(nn) } {incr i}{

   node_(i) namattach $namfile

}

for {set i 0} {i < opt(nn)} {incr i} {

   ns_ initial_node_pos node_($i) 20

}

for {set i 0} {i < opt(nn)} {incr i} {

       ns_at opt(stop).0 "node_(i) reset"

}


ns_ at opt(stop) "stop"

ns_ at opt(stop).01 "puts\"NSEXITING...\" ; $ns_ halt"

proc stop {} {

       globalns_ tracefd namfile

       $ns_flush-trace

       close$tracefd

       close$namfile

       exit0

}

puts tracefd "M 0.0 nn opt(nn) xopt(x) y opt(y) rp $opt(rp)"

puts tracefd "M 0.0 prop opt(prop)ant $opt(ant)"

puts "Starting Simulation..."

$ns_ run

```

我们在此目录中打开终端，输入

> ns gpsr-wireless.tcl

如图所示
![这里写图片描述](http://img.blog.csdn.net/20180124093906171?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
运行nam查看仿真状况

如图我们可以看到节点移动情况
![这里写图片描述](http://img.blog.csdn.net/20180124093948738?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

###trace文件分析

下面是整个仿真过程最重要的部分，trace文件分析

投递率我使用的是grep工具

>  grep “r.*AGT” trace.tr > g_r
>
>  grep “s.*AGT” trace.tr > g_s
>
>  wc g_?

即可得到数据包的发送和接收情况：
![这里写图片描述](http://img.blog.csdn.net/20180124094042085?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
然后，我们通过awk脚本来获取协议中的时延情况：

脚本代码如下：

```tcl
#BEGIN表明这是程序开头执行的一段语句，且只执行一次。

BEGIN {

#程序初始化，设定一变量以记录目前处理的封包的最大ID号码。在awk环境下变量的使用不需要声明，直接赋值。

highest_uid = 0;

}

#下面大括号里面的内容会针对要进行处理的记录（也就是我们的trace文件）的每一行都重复执行一次

{

event = 1; #1表示一行的第一栏，是事件的动作。每一栏默认是以空格分隔的。下同。

time = $2; #事件发生的时间

node_nb = $3; #发生事件的节点号（但是两边夹着“”，下面一句代码将“”处理掉）

node_nb=substr(node_nb,2,1); #第三栏的内容是形如0的节点号码，我只要得出中间的节点号码0，所以要对字符串0进行处理。

trace_type = $4; #trace文件跟踪事件的层次（指在路由层或mac层等等） 

flag = $5; #

uid = $6; #包的uid号码（普通包头的uid）

pkt_type = $7; #包的类型（是信令或是数据）

pkt_size = $8; #包的大小（byte）

#下面的代码记录目前最高的CBR流的packet ID，本来的延迟分析是针对所有的包的（包括信令），这里作了简化，只针对CBR封包，以后大家做延时分析可以做相应的改动即可。

if ( event=="s" &&node_nb==0 && pkt_type=="cbr" && uid > highest_uid)

{#if判断句的前三个判断条件就不说了，第四个是说每个包的记录次数不超过1

highest_uid = uid;

}

#记录封包的传送时间


if ( event=="s" &&node_nb==0 && pkt_type=="cbr" && uid==highest_uid )

start_time[uid] = time; # start_time[]表明这是一个数组

#记录封包的接收时间

if ( event=="r" &&node_nb ==2 && pkt_type=="cbr" && uid==highest_uid )

end_time[uid] = time;

}

#END表明这是程序结束前执行的语句，也只执行一次

END {

#当每行资料都读取完毕后，开始计算有效封包的端到端延迟时间。

for ( packet_id = 0; packet_id <=highest_uid; packet_id++ )
{

start = start_time[packet_id];

end = end_time[packet_id];

packet_duration = end - start;

#只把接收时间大于传送时间的记录打印出来

if ( start < end ) printf("%d%f\n", packet_id, packet_duration);

}

}
```

我们在终端输入命令：

>  gawk -f delay.awk trace.tr > delay.csv

因为我本人熟悉Python语言，所以绘图我就用Python和matplotlib库来做了：

```python
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

col = ['x','y']

data = pd.read_csv("delay.csv",names=col)

x = data['x']

y = data['y']

plt.xlabel('Time(s)')

plt.ylabel('Transmission Speed(KB/s)')

plt.title('GPSR Analysis')

plt.xlim(50,150)

plt.ylim(0.0015,0.0055)

plt.plot(x0,y0,color='blue',linewidth=1.5,linestyle="-",label='GPSR')

plt.plot(x,y,color='red',linewidth=1.5,linestyle="-",label='NNGPSR')

plt.show()
```

得到时延的图
![这里写图片描述](http://img.blog.csdn.net/20180124094135962?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

至此，本次ns2实验就结束了。

这篇博客写的比较匆忙，如有错误，可评论区加以说明。

----------


##参考文章

1.     [NS2笔记八gpsr移植]( <http://blog.sina.com.cn/s/blog_414c2a9a01013r9e.html>)

2.     [关于802.11p和场景文件]( <http://blog.sina.com.cn/s/blog_64ab4ef30101d9j3.html>)

3.     [NS2中cbrgen和setdest的使用]( <http://blog.sina.com.cn/s/blog_61893e6101018ad6.html>)

4.     [GPSR源码修改]( <http://blog.sina.com.cn/s/blog_64ab4ef30101d9i7.html>)

5.     [ubuntu14.04LTS下搭建NS2实验环境](<https://www.cnblogs.com/licuncun/p/5304232.html>)


