---
title: GPSR协议概述
categories: 
- note
tags: 
- GPSR
copyright: true
---

# 1.   GPSR协议简介

&emsp;&emsp;GPSR通过应用邻居节点和终点的地理位置，允许每个节点对全局路由分配做出决策。当一个节点以贪婪算法转发一个包时，它有比自己更接近终点的邻居节点，这个节点就选择距离终点最近的邻居节点来转发该包。当没有这种邻居节点时，数据包进入周围模式，将包向前传送给网络平面字图的临近节点，直到传到距离终点较近的节点，将包转发的方式为贪婪算法模式。

# 2.   GPSR协议流程

![这里写图片描述](http://img.blog.csdn.net/20180124204504143?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzA2MTE2MDE=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 3.   协议源文件

> gpsr_packet.h : 定义不同类型的包
>
> gpsr_neighbor.h : 定义该gpsr实现所使用的每个节点的邻居列表
>
> gpsr_neighbor.cc : 邻居列表类的实现
>
> gpsr.h : 该实现的GPSR路由代理函数的定义
>
> gpsr.cc : GPSR路由代理的实现
>
> gpsr_sinklist.h: 用于多个接收器的场景的定义
>
> gpsr_sinklist.cc: 实现gpsr_sinklist.h

 

# 4.   宏定义

```c++
#define DEFAULT_GPSR_TIMEOUT   200.0		//生存时间

#define INIFINITE_DISTANCE   1000000000.0		//无穷大

 

#define SINK_TRACE_FILE "sink_trace.tr"			//sink_trace文件

#define NB_TRACE_FILE "gpsrnb_trace.tr"			//nb_trace文件

#define GPSR_CURRENT Scheduler::instance().clock()		//计时器

#define INFINITE_DELAY 5000000000000.0			//无穷大

#define GPSRTYPE_HELLO  0x01   //hello msg

#define GPSRTYPE_QUERY  0x02   //query msg from the sink

#define GPSRTYPE_DATA   0x04   //the CBR data msg

#define GPSR_MODE_GF    0x01   //greedy forwarding mode

#define GPSR_MODE_PERI  0x02   //perimeter routing mode

#define HDR_GPSR(p)   ((structhdr_gpsr*)hdr_gpsr::access(p))	//gpsr报头

#define HDR_GPSR_HELLO(p) ((struct hdr_gpsr_hello*)hdr_gpsr::access(p))		//hello报头

#define HDR_GPSR_QUERY(p) ((struct hdr_gpsr_query*)hdr_gpsr::access(p))		//query报头

#define HDR_GPSR_DATA(p) ((struct hdr_gpsr_data*)hdr_gpsr::access(p))		//data报头

 

#define PI 3.141593			//PI

#define MAX(a, b)(a>=b?a:b)		//最大

#define MIN(a, b)(a>=b?b:a)		//最小

```



# 5.   结构体

```c++
struct hdr_gpsr 				//gpsr报头

struct hdr_gpsr_hello 			//hello报头

struct hdr_gpsr_query 			//query报头

struct hdr_gpsr_data			//data报头

union hdr_all_gpsr				//总报头

 

struct gpsr_neighbor			//邻居

 

struct sink_entry				//数据接收器

```

 

# 6.   类

```c++
class Sinks            //sink表维护一个数据接收器列表，它用于多个数据接收器，这不是GPSR设计的一部分

class GPSRNeighbors     //网络中每个节点的邻居列表

class GPSRAgent        //GPSR路由代理，定义路由代理、路由的方法(行为)

class GPSRHelloTimer: public TimerHandler     //定时发送‘hello’信息

class GPSRQueryTimer: public TimerHandler     //数据接收器使用的查询计时器来触发数据查询。它不是GPSR路由设计的一部分。

```

 

# 7.   相关函数

```c++
void GPSRHelloTimer::expire(Event*e)       //hello计时器计时方法

void GPSRQueryTimer::expire(Event*e)       //查询计时器计时方法

void GPSRUpdateSinkLocTimer::expire(Event *e)   //数据接收器更新计时器计时方法

void GPSRAgent::hellotout()                //侦查函数

void GPSRAgent::updatesinkloctout()        //数据接收器侦查

void GPSRAgent::startSink()                //开始接受数据

void GPSRAgent::startSink(doublegp)        //开始接受数据

void GPSRAgent::querytout()            //查询侦查

void GPSRAgent::getLoc()               //获取位置

void GPSRAgent::GetLocation(double *x, double *y)   //获取位置

GPSRAgent::GPSRAgent(): Agent(PT_GPSR), 

            hello_timer(this), query_timer(this),

         update_sink_loc_timer_(this),

            my_id(-1), my_x(0.0), my_y_(0.0),

            recv_counter(0), query_counter(0),

            query_period_(INFINITE_DELAY)		//协议初始化

void GPSRAgent::turnon()       //开启协议

void GPSRAgent::turnoff()      //关闭协议

void GPSRAgent::hellomsg()     //发送hello包

void GPSRAgent::query(nsaddr_t id)          //开始查询

void GPSRAgent::recvHello(Packet *p)        //接受hello包

void GPSRAgent::recvQuery(Packet*p)        //接受查询信息

void GPSRAgent::sinkRecv(Packet *p)         //数据接收器接受包信息

void GPSRAgent::forwardData(Packet*p)      //数据信息判断并转发

void GPSRAgent::recv(Packet *p, Handler *h) //接受数据包

void GPSRAgent::trace(char *fmt, ...)           //trace函数

int GPSRAgent::command(int argc, const charconst argv)        //接受参数命令

 


GPSRNeighbors::GPSRNeighbors()         //邻居表初始化

GPSRNeighbors::~GPSRNeighbors()        //析构函数

double GPSRNeighbors::getdis(double ax, double ay, double bx, double by)    //获取位置

int GPSRNeighbors::nbsize()                //邻居节点数量

void GPSRNeighbors::myinfo(nsaddr_t mid, double mx, double my)      //获取信息

struct gpsr_neighbor* GPSRNeighbors::getnb(nsaddr_t nid)               //获取邻居表

void GPSRNeighbors::newNB(nsaddr_t nid, double nx, double ny)           //添加新的邻居节点

void GPSRNeighbors::delnb(nsaddr_t nid)         //删除邻居节点

void GPSRNeighbors::delnb(struct gpsr_neighbor *nb)     //删除邻居节点

void GPSRNeighbors::delalltimeout()            //删除所有超时节点信息

nsaddr_t GPSRNeighbors::gf_nexthop(double dx, double dy)        //贪婪模式下一跳

struct gpsr_neighbor* GPSRNeighbors::gg_planarize()           

struct gpsr_neighbor* GPSRNeighbors::rng_planarize()          //进行周长路由计算

double GPSRNeighbors::angle(double x1, double y1, double x2, double y2) //计算角度

int GPSRNeighbors::intersect(nsaddr_t theother, double sx, double sy,

             double dx, double dy)              //检查两条线是否局部相交

int GPSRNeighbors::num_of_neighbors(struct gpsr_neighbor *nblist)       //给定邻居表的节点数量

void GPSRNeighbors::free_neighbors(struct gpsr_neighbor *nblist)        //释放邻居节点

nsaddr_t GPSRNeighbors::peri_nexthop(inttype_, nsaddr_t last,

               double sx, double sy,double dx, double dy)          //周边模式下一跳

void GPSRNeighbors::dump()     //转储邻居表

 

 

Sinks::Sinks()             //数据接收器初始化

bool Sinks::new_sink(nsaddr_tid, double x, double y, 

        nsaddr_t lasthop, int hops, int seqno)             //创建新的数据接收器

bool Sinks::remove_sink(nsaddr_t id)                       //删除数据接收器

void Sinks::getLocbyID(nsaddr_t id, double &x, double &y, int &hops)        //通过ID获取节点位置

void Sinks::dump()         //转储

```




