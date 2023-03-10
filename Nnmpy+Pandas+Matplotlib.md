 数据分析

[文字教程](https://www.gairuo.com/p/numpy-tutorial)

[官网](https://matplotlib.org/)

---

### Numpy

#### 1.  基础

>*metadata + data*
>
>```python
>import numpy as np# 导入 numpy 库, 约定俗成别名为 np
>```
>
>+ 创建数据
>
>```python
>np.array([1, 2, 3])
>np.array((1, 2, 3)) # 同上
># array([1, 2, 3])
>
>np.array(((1, 2),(1, 2)))
>np.array(([1, 2],[1, 2])) # 同上
># out:
>array([[1, 2],
>        [1, 2]])
>
>np.arange(10) # 10个, 不包括10，步长为 1
>np.arange(3, 10, 0.1) # 从 3 开始到10，步长为 0.1
># 从 2.0 开始到 3.0，生成均匀的 5 个值，不包括终终值 3.0
>np.linspace(2.0, 3.0, num=5, endpoint=False)
> # 返回一个 6x4 的随机数组，float 型
>np.random.randn(6, 4)
># 指定范围指定形状的数组，整型
>np.random.randint(3,7,size=(2, 4))
># 创建从0到20随机的5个数组
>np.random.randint(0, 20, 5)
># array([ 9, 10, 14,  6, 14])
># 创建值为 0 的数组
>np.zeros(6) # 6个浮点 0.
>np.zeros((5, 6), dtype=int) # 5 x 6 整型 0
>np.ones(4) # 同上
>np.empty(4) # 同上
> # 创建一份和目标结构相同的 0 值数组
>np.zeros_like(np.arange(6)) # [0 0 0 0 0 0]
>np.ones_like(np.arange(6)) # 同上
>np.empty_like(np.arange(6)) # 同上
>np.arange(4).repeat(4) # 将4个值依次重复四次，共16个
>np.full((2,4), 9) # 两行四列值全为 9
>```
>
>+ 数据类型
>
>```python
>np.int64 # 有符号 64 位 int 类型
>np.float32 # 标准双精度浮点类型
>np.complex # 由128位的浮点数组成的复数类型
>np.bool # TRUE 和 FALSE 的 bool 类型
>np.object # Python 中的 object 类型
>np.string # 固定长度的 string 类型
>np.unicode # 固定长度的 unicode 类型
>
>np.NaN # np.float 的子类型
>np.nan
>
># np 所有数据类型
>np.sctypeDict # np.typeDict 将弃用
>```
>
>+ 数组信息
>
>```python
>n.shape # 数组的形状, 返回值是一个元组
>n.shape = (4, 1) # 改变形状
>a = n.reshape((2,2)) # 改变原数组的形状创建一个新的
>n.dtype # 数据类型
>n.ndim # 维度数
>n.size # 元素数
>a = np.array([1,2,3,4,5,6,])
>b = a.astype(float) # 转换array类型
>```
>
>+ 计算
>
>```python
>np.array( [10,20,30,40] )[:3] # 支持类似列表的切片
>a = np.array( [10,20,30,40] )
>b = np.array( [1, 2, 3, 4] )
>a+b # array([11, 22, 33, 44]) 矩阵相加
>a-1 # array([ 9, 19, 29, 39])
>4*np.sin(a)
>
># 以下举例数学函数，还支持非常多的数据函数
>a.max() # 40
>a.min() # 10
>a.sum() # 100
>a.std() # 11.180339887498949
>a.all() # True
>a.cumsum() # array([ 10,  30,  60, 100])
>b.sum(axis=1) # 多维可以指定方向
>np.negative(-1) # 1 相反数
>```
>
>+ 逻辑
>
>```python
>x = 10
>np.where(x>0, 1, 0) # array(1)
>np.logical_and(x>0, x>5) # True
>np.logical_or(x>0, x<5) # True
>np.logical_not(x>5) # False
>np.logical_xor(x>5, x==0) # 异或 True
>```
>
>
>
>+ 集合
>
>```python
># 输出x和y的交集
>np.intersect1d(x,y)
># 输出在x中不在y中的元素
>np.setdiff1d(x,y)
># 输出x和y的并集
>np.union1d(x,y)
># 输出x和y的异或集
>np.setxor1d(x,y)
>```
>
>
>
>+ 合并
>
>```python
># 以下方法传入的是一个 *arrays，如 (a1, a2, ...)
>np.append() # 追加, 可读性好占内存大
>np.stack()
>np.dstack()
>np.vstack()  # 垂直合并
>np.hstack() # 水平合并
>np.newaxis() # 转置
>np.concatenate() # 针对多个矩阵或序列的合并操作
>np.split(ab, 4, axis=0) # 分隔
>```

#### 2. 数据类型

>+ Data type objects (dtype)
>
>```python
>dt = np.dtype('>i4')
>dt.byteorder
># '>'
>dt.itemsize
># 4
>dt.name
># 'int32'
># 对应的数组标量类型是 int32
>dt.type is np.int32
># True
>np.float64 == np.dtype(np.float64) == np.dtype('float64')
># True
>np.float64 == np.dtype(np.float64).type
># True 
>```
>
>+ 设置ndarry
>
>```python
># 1
>data = [('zs', [11,22,33], 19)，
>       ('ls',[33,44,55],222)
>        ('ww',[55,66,77], 22)]
>a = np.array(data. dtype='U3,3int32, int32 )
>#2
>a = np.array(data, dtype = [('name','str',2),
>            ('scores','int32',3),
>            ('age','int32',1)] )
>#3
>a= np.array(data, dtype = {
>    'names': ['name''scores' 'ages'],
>    'format':['U3','3int32','int32']})
>a[2]['age']
>#4
>dt = np.dtype([('name', np.unicode_, 16), ('grades', np.float64, (2,))])
>x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
>
>```
>
>+ 日期
>
>```python
>datas = ['2011-01-01','2011','2011-02','2012-01-01','2011-02-01 10:10:00',]
>datas = np.array(dates)
>dates = dates.astype('M8[D]') 
>#精确到日 'D'  精确到秒 'S' 不足自动填充
>dates - dates
>```
>
> np.int8 ->i1  8 digit = 1 byte
>
>| 字符 | 说明                    |
>| ---- | ----------------------- |
>| ?    | 布尔型 bool             |
>| b    | （有符号）字节          |
>| B    | 无符号字节              |
>| i    | (有符号) 整型 int       |
>| u    | 无符号整型 integer uint |
>| f    | 浮点型 float            |
>| c    | 复数浮点型 complex      |
>| m    | timedelta（时间差）     |
>| M    | datetime（日期时间）    |
>| O    | (Python) 对象           |
>| S, a | (byte-)字符串（不推荐） |
>| U    | Unicode 字符串 str      |
>| V    | 原始数据 (void)         |
>
> 
>
>| dtype.type | 用于实例化此数据类型的标量的类型对象                    |
>| ---------- | ------------------------------------------------------- |
>| dtype.kind | 一种字符码（“biufcmMOSUV”之一），用于识别一般类型的数据 |
>| dtype.char | 为21种不同的内置类型中的每种类型提供唯一的字符代码      |
>| dtype.num  | 21种不同内置类型中每种类型的唯一编号                    |
>| dtype.str  | 此数据类型对象的数组协议 typestring                     |
>
>   
>
>| 属性           | 说明                     |
>| -------------- | ------------------------ |
>| dtype.name     | 此数据类型的位宽度名称   |
>| dtype.itemsize | 此数据类型对象的元素大小 |
>
> 
>
>| 属性            | 说明                               |
>| --------------- | ---------------------------------- |
>| dtype.byteorder | 表示此数据类型对象的字节顺序的字符 |

#### 3. 维度操作

>+ view 数据共享: reshappe() & ravel() 与原数组使用同一份数据
>
>```python
>a = np.arange(1,9)
>b = a.reshape(2,3)
>c = b.reshape(2,2,2) # 2*2*2 三维
>d = c.ravel() # 变为一维数组
>```
>
>+ 复制变维(数据独立): flatten()
>
>	```python
>	e = c.faltten() # 变为一维数组 
>	```
>
>	
>
>+ 直接改
>
>```python
>c.shape = (3, 3)
>c.resize((9,)) # 一行九列
>```

#### 4. 索引&切片

>```python
> a = np.arange(1,9)
> a[-4:-7:-1] # 倒着切片 倒着显示
>a.resize(3,3)
>z[:2,:2] # 二维 前两行前两列 ","分割维度
>```

#### 5. ndarry 数组掩码操作

>```python
># bool 
>a = np.arange(1,10)
>mask = [True, True, False, False, False, False, False, False, False,]
>a[mask] # (1,2)
>a = np.arrange(1,101)
>a(a%3==0) # 100以内3的倍数 a%3==0 返回true or false
>a((a%3==0) & (a%7==0)) #用 & 不能 and
># 索引
>names = np.array(['apple', 'samsung', 'Xiaomi', 'Vivo', 'Huawei'])
>rank = [0,1,3,4,2]
>names[rank]
># ['samsung' 'apple' 'Vivo' 'Huawei' 'Xiaomi']
>```
>

#### 6. 多维数组组合拆分

>```python
>a = np.arange(1,7).reshape(2,3)
>b = np.arange(7,23).reshape(2,3)
>c = np.vstack((a,b)) #垂直
>d,e = np.vsplit(c,2)
>c = np.hstack((a,b)) #水平
>d,e = np.hsplit(c,2)
>c = np.dstack((a,b)) # 深度
>d,e = np.dsplit(c,2)
>np.concatenate((a,b),axis=0)  # 0垂直 1水平 2深度（a.b 必须三维）
>np.row_stack((a,b)) # 两行  
>np.column_stack((a,b)) #两列
>```

#### 7. 长度不等

>```python
>a = np.array([1,2,3,4,5])
>b = np.array([1,2,3,4])
>b = np.pad(b, pad_width=(0,1),mode='constant',constant_values=-1)
>```
>

#### 8. 加载文件

>```python
>np.loadtxt(
>'/xxx.csv',#路径
>deilmiter=','#分隔符 两个对象以','分隔
>usecols=(1,3),#读取1，3 列 下标从0开始
>unpack=False,#是否按列拆包 是第一列第三列拆开两个数组返回 还是合在一起数组返回  拆开要有两个接受
>dtype='U10,F8',#制定返回每一列数组中元素的类型
>converters={1:func})#转换器函数字典 下标为1 的列 走函数func 
>```

#### 9. 转换日期

>```python
>dates,prices=np.loadtxt(
>'/xxx.csv',
>deilmiter=','
>usecols=(1,3),
>unpack=True, 
>dtype='M8[D],F8',
>converters={1:dmy2ymd})# numpy 日期只支持年月日 不支持日月年 用转换器转
>```
>
>```python
>def dmy2ymd(dmy):
>    dmy= str(dmy,encoding='utf-8')
>    time = dt.datetime.steptime(dmy,'%d-%m-%Y').date()
>    t = time.strftime('%Y=%m=%d')
>    return t
>```
>
>```python
>plt.figure('xxx',facecolor='lightgray')
>plt.title('xxx',fontsize=16)
>plt.xlabel('date',fontsize=14)
>plt.ylabel('prise',fontsize=13)
>plt.grid(linstyle=':')
>import matplotlib.dates as md
>ax = plt.gca()
>ax.xaxis.set_major_locator(md.WeekdayLocator(byweekend=md.MO))  #每周一MO
>ax.xaxis.set_major_locator(md.DateFormatter(%d %b %Y))
>ax.xaxis.set_minor_locator(md.DayLocator())
>dates = dates.astype(md.datetime.datetime)
>plt.plot(dates,prise,color='dodgerblue',label='xxx',linestyle='--',linewidth=2)
>plt.legend()
>plt.gcf()autofmt_xdate()
>plt.show()
>
>```
>
>
>
>

### matplotlib

---

#### 1. 基础

>```python
>import matplotlib.pyplot as plt
>import numpy as np
>x = np.array([1,2,3,4,5,6])
>y = np.array([12,34,45,67,78,45])
>plt.plot(x,y)
>plt.show()
>```
>
>```python
>plt.vlines(vval,ymin,ymax,...) #垂直线
>plt.vlines([],[],[],   ...)
>ply.hlines(xval,xmin,xmax,...) #水平线
>```
>
>```python
>x = np.linspace(-np.pi,np.pi,1000)
>sinx = np.sin(x)
>cosx = np.cox(x)
>plt.plot(x,sinx)
>plt.plot(x,cosx)
>plt.show()
>```
>
>```python
>plt.plot(xarray,yarray,linestyle='', linewidth=1, color= '', alpha=0.5)
># alpha 透明度 线形 linestyle：'-' '--' '-.' ':'
>```
>
>```python
>plt.xlim(x_limt_min,x_limit_max) #坐标轴可视
>plt.ylim(y_limt_min,y_limit_max)
>```
>
>```python
>plt.xticks(x_val_list,x_text_list) #坐标轴
>plt.yticks(y_val_list,y_text_list)
>```
>
>```python
># latex 语法
>r'$pi$'   -> Π (3.1415926)
>```
>
>```python
>#坐标轴
>ax = plt.gca() # {'left’:左 right bottom top}
>axis = ax.apines['坐标轴名']
>axis.set_position((type,val))
>axis.set_color(color)
>#example 坐标系移到(0，0)
>ax = plt.gca()
>ax.spines['top'].set_color('none')
>ax.spines['right'].set_color('none')
>ax.spines['left'].set_position(('data',0))
>ax.spines['bottom'].set_position(('data',0))
>```
>
>```python
>plt.legend(loc='') #  location string location code (best 0)  (upper right 1)  (lower right 4) (center right 7) (center 10)
>plt.plot(x,y1,label='sdf')
>plt.plot(x,y2,label='ccxb')
>plt.legend()
>```
>
>```python
># 绘制特殊点
>plt.scatter(xarray,yarray,
>      marker='',#点型
>      s='',#大小
>           label=''#标签
>           edgecolor='',#边缘色
>           facecolor='',#填充色
>           zorder=3#绘制图层编号
>           )
>     ```
>     
>```python
>#备注
>plt.annotate(
>           '内容'，#备注中显示的内容
>             xycoorder='data', #备注目标所使用的坐标系 data表示数据坐标系
>                 xy=(x,y), #坐标
>                 textcoorder='offser points', #文本的坐标 offset points 参照点的偏移坐标系
>                 xytext=(x,y),#文本坐标
>                 fontsize=14,#文本字体
>                 arrowprops=dict()#箭头样式 
>                 )
>     #arrowprops 字典参数常用key
>     arrowprops=dict(
>    arrowstyle='->', #箭头样式
>    connectionstyle='angle3'#连接线样式
>    )
>    ```
>    

#### 2.图形窗口

>```python
>plt.figure(
>'', #标题栏文本
>figsize =(4,3) # 窗口大小
>dpi=120, # 像素密度
>facecolor='' # 图表背景色
>)
>#调用一次创建一张图 应用在当前窗口
>plt.figure('a figure',facecolor='red')
>plt.plot([1,2],[1,2])
>plt.figure('b figure',facecolor='blue')
>plt.plot([2,3],[4.5])
>plt.figure('a figure',facecolor='red')
>plt.plot([3,2],[6,2])# 想继续画 可以在调一次 
>```
>
>```python
>plt.title(title,fontsize=12) # 图表标题
>plt.xlabel(x_label_str,fontsize=12) # 水平轴文本
>plt.ylabel(y_label_str,fontsize=12) # 垂直轴文本
>plt.tick_params(...,labelseze=8)#设置刻度文本参数大小
>plt.grid(linestyle='')#图标上网格线
>plt.tight_layout()#紧凑布局 显示有问题了 字与字叠加 可以调用一下
>```

#### 3. 子图

>###### 1.矩阵式布局
>
>```python
>plt.figure('subplot layout',facecolor='gray')
>plt.subplot(row,cols,num)
># 1 2 3
># 4 5 6
># 7 8 9
>plt.subplot(3,3,5)  #分成 三行三列 画在五号上
>plt.subplot(335)
>plt.text(
>0.5,0.5,1, #文本位置 内容
>ha='center',  #垂直 水平居中
>va='center',
>size=36, #字体大小
>alpht=0.5,#透明度
>withdash=False #破折号
>)
>plt.xticks([])
>plt.yticks([])
>plt.tight_layout()
>#for 循环版本
>for i in range(1,10):
>    plt.subplot(3,3,1)  #分成 三行三列 画在五号上
>    #plt.subplot(335)
>    plt.text(
>    0.5,0.5,i, #文本位置 内容
>    ha='center',  #垂直 水平居中
>    va='center',
>    size=36, #字体大小
>    alpht=0.5,#透明度
>    withdash=False #破折号
>    )
>    plt.xticks([])
>    plt.yticks([])
>    plt.tight_layout()
>```
>
>###### 2. 网格式布局
>
>```python
>import matplotlib.gridspec as mg
>plt.figure('grid layout',facecolor='lightgray')
>gs= plt.Gridspec(row,cols)
>gs=plt.Gridspec(3,3) #拆分3*3
>plt.subplot(gs[0,：2])#合并0行的0，1列 
>plt.text(0.5,0.5,'1',ha='center',va='center',size=36)
>plt.xticks([])
>plt.yticks([])
>plt.tight_layout()
>
>plt.subplot(gs[:2,2])#合并2列的1，2行
>plt.text(0.5,0.5,'2',ha='center',va='center',size=36)
>plt.xticks([])
>plt.yticks([])
>plt.tight_layout()
>```
>
>###### 3. 自由式布局
>
>```python
>plt.figure('folw layout',facecolor='gray')
>#left_bottom_x 左下角x坐标 偏移量
>#width 宽 height 高
>plt.axes([left_bottom_x,left_bottom_y,width,height])
>plt.axes([0.03,0.03,0.94,0.5]) #0.92 是占比
>plt.text(0.5,0.5,1,ha='center',va='center',size=36)
>
>plt.axes([0.03,0.5,0.94,0.5]) # 同一张图里插入第二张子图 直接写
>plt.text(0.5,0.5,1,ha='center',va='center',size=36)
>
>```

#### 4. 刻度定位器

>```python
>ax= plt.gca()#获取当前坐标轴
>ax.xaxis.set_major_locator(plt.Nulllocator())#水平坐标轴的主刻度定位器
>ax.xaxis.set_minor_locator(plt.Multiplelocator(0.1))#水平坐标轴次刻度定位器，间隔0.1
>```
>
>```python
>plt.figure('locators',facecolor='gray')
>plt.xlim(1,10)
>ax = plt.gca()
>ax.spines['top'].set_color('none')
>ax.spines['left'].set_color('none')
>ax.spines['right'].set_color('none')
>ax.spines['bottom'].set_position(('data',0.5))
>plt.yticks([])
>ax.xaxis.set_major_locator(plt.MultipleLocator(1))
>ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
>```
>
>

#### 5. 刻度网格线

>
>```python
>ax.grid(
>which='',# major minor 主次刻度
>axis='', # x y both 轴
>linewidth=1, # 线宽线性 颜色透明
>linestyle='', 
>color='',
>alpha=0.5)
>```
>
>```python
>plt.figure('grid line', facecolor='gray')
>ax= plt.gca()
>
>ax.xaxis.set_major_locator(plt.MultipleLocator(1))
>ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
>ax.yaxis.set_major_locator(plt.MultipleLocator(200)) 
>ax.yaxis.set_minor_locator(plt.MultipleLocator(50))
>
>ax.grid(wchhi='major',axis='both',color='orangered',linewidth=0.75)
>ax.grid(wchhi='ninor',axis='both',color='orangered',linewidth=0.25)
>
>y = [1,12,122,1222,222,22,2]
>plt.plot(y,'o-',color='dofgerblue')
>plt.show
>```

#### 6. 半对数坐标

y轴以指数递增

>```python
>plt.figure('grid',facecolor='lightgray')
>y=[1,11,111,1111,111,11,1]
>plt.senilogy(y,'o-',color='dodgerblue')
>plt.show
>```

#### 7. 散点图

>```python
>plt.scatter(
>x, #x y 轴坐标数
>y,
>marker='', # 点型
>s=10, # 大小
>color=''
>edgecolor=''
>facecolor=''
>zorder='')
>```
>
>```python
>n = 100
>height = np.random.normal(172,20,n) # 期望 标准差 生成数量
>weight = np.random.normal(60,10,n)
>
>plt.figure('scatter',facecolor='lightgray')
>plt.title('persons',fontsize=18)
>plt.xlabel('height',fontsize=14)
>plt.ylabel('weight',fontsize=14)
>plt.grid(linestyle=':')
>d = (height-175)**2+(weight-70)**2
>plt.scatter(height,weight,marker='o',s=70,label='persons',c=d,cmap = 'jet')# c 颜色 离标准约远 颜色改变大
>plt.legend()
>plt.show

#### 8. 填充

>```python
>mp.fill_between(
>x,#坐标
>sin_x,#下边界垂直坐标
>cos_x,#上边界垂直坐标
>sinx<cos_x,#填充条件 True时填充
>color='',
>alpha=0.2)
>```
>
>```python
>x = linespace(0,8*np.pi,1000)
>sinx = np.sin(x)
>cosx = np.cos(x/2)/2
>plt.figure('fill',facecolor='lightgray')
>plt.title('fill',fontsize=18)
>plt.grid(linestyle=':')
>plt.plot(x,sinx,color='dodgerblue',label=r'$y=sin(x)$' )
>plt.plot(x,cosx,color='orangered',label='$y=\frac{1}{2}cos(\frac{x}{2})$')
>plt.fill_between(x,sinx,cosx,sinx>cosx,color='dodgerblue',alpha=0.3)
>plt.fill_between(x,sinx,cosx,sinx<cosx,color='orangered',alpha=0.3)# sinx cosx可换位置
>plt.legendd()#图例
>```

#### 9. 条形图

>```python
>plt.figure('bar',facecolor='lightgray')
>plt.bar(
>x,
>y,
>width,
>color='',
>label='',
>alpha=0.2
>)
>```
>
>```python
>apple = np.array([12,34,56,56,78,56,34,23,46,67,45,56])
>orange= np.array([22,24,66,86,98,66,54,33,26,17,35,66])
>plt.figure('bar',facecolor='ligghtgray')
>plt.title('bar chart',fontsize=18)
>plt.grid(linestyle=':')
>x = np.arange(apple.size)
>plt.bar(x-0.2,apple,0.4,color='red',label='apple')
>plt.bar(x+0.2,orange,0.4,color='orange',label='orange')
>'''
>plt.bar(x,apple,0.8,color='red',label='apple')
>plt.bar(x,orange,0.8,color='orange',label='orange')
>'''
>plt.xticks(x,['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
>plt.legend()
>plt.show()
>
>```

#### 10.饼图

>```python
>plt.pie(
>[values],#值列表
>[spaces],#扇形之间间距列表
>[labels],#标签列表
>[colors],#颜色列表
>'%d%%',#标签所占比例格式 %d->整数 %%->%  %.1f%%
>shadow=True,#阴影
>startangle=90#逆时针饼图 起始角度
>radius=1)#半径
>plt.axis('equal')#等轴比例
>```

#### 11. 等高线图

>```python
>plt.contour(x,y,x,8,color='black',linewidth=0.5)
>cntr = plt(
>x,#x,y,z 都是二维数组
>y,
>z,
>8,#等高线绘制成8部分
>color='',#等高线颜色
>linewidth=0.5)
>plt.clabel(cntr,
>           inline_spacing=1, #间隙边距
>           fmt='%.1f',# format
>           fontsize=10)
>plt.contourf(x,y,z,8,cmap='jet') #f->fill 填充
>```
>
>```python
>n = 1000
>x,y = np.meshgrid(np.linspace(-3,3,n),
>            np.linspace(-3.3.n))
>z = (1-x/2+x**5+y**3)*\np.exp(-x**2-y**2) # 根据xy算z 高度
>```
>
>```python
>plt.figure('contour',facecolor='lightgray')
>plt.title('contour',fontsize=16)
>plt.grid(linestyle=':')
>cntr = plt,contour(x,y,z,8,color='black',linewidths=0.5)
>plt.clabel(cntr,fmt='%.2f',inline_spacing=2,fontsize=8) #设置等高线高度标签文本 cntr
>plt.contourf(x,y,z,8,cmap='jet')
>plt.show()
>```

#### 12.热成像图

>图像显示矩阵和中止大小
>
>```python
># orgin 坐标轴方向
>#upper 缺省值 远点左上角
>#lower 远点在左下角
>plt,imshow(z,cmap='jet',origin='low')
>```
>
>```python
>x,y = np.meshgrid(np.linspace(-3,3,n),
>            np.linspace(-3.3.n))
>z = (1-x/2+x**5+y**3)*\np.exp(-x**2-y**2) # 根据xy算z 高度
>plt.imshow(z,cmap='jet',origin='lower')
>plt.colorbar() #旁边显示个颜色渐变条
>plt.show()
>```

#### 13. 3D图像绘制

>
>
>```python
>from mpl_toolkits.mplot3d import axes3d
>ax3d= plt.gca(projection='3d')
>ax3d.scatter(..)
>ax3d.plot_surface(..)
>ax3d.plot_wireframe(..)
>```
>
>```python
>ax3d.scatter(
>x,
>y,
>marker=''.#点型
>s=10, #大小 size
>zorder='',
>color='',
>edgercolor='',
>facecolor='',
>c=v,#颜色值 根据cmap 映射应用相应的颜色
>cmap='')
>```
>
>```python
>n = 300
>x= np.random.normal(0,1,n)
>y= np.random.normal(0,1,n)
>z= np.random.normal(0,1,n)
>plt.figure('3d project',facecolor='lightgray')
>ax3d = plt.gca(projection='3d')
>ax3d.set_xlabel('X')
>ax3d.set_ylabel('Y')
>ax3d.set_z label('Z')
>d= x**2+y**2+z**2
>ax3d.scatter(x,y,z,s=60,marker='o',c=d,cmap='jet_r')
>plt.show()
>```
>
>```python
>#三维曲面
>ax3d.plot_surface(
>x,
>y,
>z,
>rstride=30,#行列跨距  越小图片越细腻
>cstride=30,
>cmap='jet')
>```
>
>```python
>n = 300
>x= np.random.normal(0,1,n)
>y= np.random.normal(0,1,n)
>z= np.random.normal(0,1,n)
>plt.figure('3d',facecolor='lightgray')
>ax3d = plt.gca(projection='3d')
>ax3d.set_xlabel('X')
>ax3d.set_ylabel('Y')
>ax3d.set_z label('Z')
>ax3d.plot_surface(x,y,z,cstride=30,rstride=30,cmap='jet')
>plt.show()
>```
>
>```python
>#3d线框图
>n = 300
>x= np.random.normal(0,1,n)
>y= np.random.normal(0,1,n)
>z= np.random.normal(0,1,n)
>plt.figure('3d',facecolor='lightgray')
>ax3d = plt.gca(projection='3d')
>ax3d.set_xlabel('X')
>ax3d.set_ylabel('Y')
>ax3d.set_z label('Z')
>ax3d.plot_wireframe(x,y,z,cstride=30,rstride=30,linewidth=1,color='dodgerblue')
>plt.show()
>```

#### 14.极坐标系

适合显示与角度有关的图像

>
>
>```python
>plt.figure('polar',facecolor='lightgray')
>plt.gca(projection='polar')
>plt.title('polar',fontsize=20)
>plt.xlabel=(r'$\theta$',fontsize=14)
>plt.ylabel=(r'$\rho$',fontsize=14)
>plt.tick_params(labelsize=10)
>plt.grid(linestyle=':')
>plt.show()
>#
>t = np.linepace(0,4*np.pi,1000)
>r = 0.8*t
>plt.plot(t,r)
>plt.show()
>#y=3sin(6x)
>x=np.linespace(0,6*np.pi,1000)
>y = 3*np.sin(6*x)
>```

#### 15. 简单动画

>```python
>import matplotlib.animation as ma
>def update(number):
>    pass
>#每10毫秒update执行一次 作用plt.gcf()
>#plt.gcf()获取当前窗口
>#interval间隔时间
>
>```
>
>```python
>anim = ma.FuncAnimation(plt.gcf(),update,interval=10)
>plt.show()
>```
>
>```python
>ball_type = np.dtype([
>    ('position',float,2),
>    ('size',float,1),
>    ('growth',float,1),
>    ('color',float,4)
>])  
>n = 100
>balls = np.zeros(100,dtype=ball_type)
>balls['position']= np.radom,uniform(0,1,(n,2)) #从0到1 生成n行2列
>balls['size']= np.radom,uniform(40,70,n)
>balls['growth']= np.radom,uniform(10,20,n)
>balls['color']= np.radom,uniform(0,1,(n,4))
>
>plt.figure('animation',facecolor='lightgray')
>plt.title('animation',fontsize=14)
>plt.xticks
>plt.yticks(())
>sc = plt.scatter(
>balls['position'][:,0],
>balls['position'][:,1],
>balls['size'],
>color=balls['color'],
>alpha=0.5)
>def update(number):
>        index= number%100
>        #修改index位置元素的属性
>        balls['position'][index]=\
>        np.randomuniform(0,1,(1,2))
>        balls['size'][index]= np.uniform(50,70,1)
>        balls['size']+= balls['growth']
>        #重新绘制
>        sc.set_size(balls['size'])# 更新大小
>        sc.set_offsets(balls['position'])#更新位置
>    
>anim = ma.FuncAnimation(plt.gcf(),update,interval=30) #需要返回值
>plt.show()
>```
>

#### 16. 生成器

>
>
>```python
>def update(data):
>	t,v = data
>    ...
>    pass
>def generator(): #每10 毫秒调用生成器 把数据给update 更新函数图像
>    yield y,v
>anim = ma.FuncAnimation(
>mp.gcf(),
>update,
>y_generator,
>interval=20)
>
>```
>
>```python
>def update(data):
>	t,v = data
>    x,y = pl.get_data()
>    x.append(t)
>    y.append(v)
>    pl.set_data(x,y)
>    if(x[-1]>10):
>		plt.xlim(x[-1]-10,x[-1])
>    
>def y_generator(): #每10 毫秒调用生成器 把数据给update 更新函数图像
>    global x
>    y = np.sin(2*np.pi*x)*np.exp(np.sin(0.2*np.pi*x))
>    yield (x,y)
>    x+=0.05
>anim = ma.FuncAnimation(
>mp.gcf(),
>update,
>y_generator,
>interval=20)
>```

### Practical

---

```python
def dmy2ymd(dmy):
    dmy= str(dmy,encoding='utf-8')
    time = dt.datetime.steptime(dmy,'%d-%m-%Y').date()
    t = time.strftime('%Y=%m=%d')
    return t
plt.figure('xxx',facecolor='lightgray')
plt.title('xxx',fontsize=16)
plt.xlabel('date',fontsize=14)
plt.ylabel('prise',fontsize=13)
plt.grid(linstyle=':')
import matplotlib.dates as md
ax = plt.gca()
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekend=md.MO))  #每周一MO
ax.xaxis.set_major_locator(md.DateFormatter(%d %b %Y))
ax.xaxis.set_minor_locator(md.DayLocator())
dates = dates.astype(md.datetime.datetime)
plt.plot(dates,prise,color='dodgerblue',label='xxx',linestyle='--',linewidth=2)
plt.legend()
plt.gcf()autofmt_xdate()
plt.show()

```



#### 1. K线图 38.28

>```python
>def dmy2ymd(dmy):
>    dmy= str(dmy,encoding='utf-8')
>    time = dt.datetime.steptime(dmy,'%d-%m-%Y').date()
>    t = time.strftime('%Y=%m=%d')
>    return t
>plt.figure('xxx',facecolor='lightgray')
>plt.title('xxx',fontsize=16)
>plt.xlabel('date',fontsize=14)
>plt.ylabel('prise',fontsize=13)
>plt.grid(linstyle=':')
>import matplotlib.dates as md
>ax = plt.gca()
>ax.xaxis.set_major_locator(md.WeekdayLocator(byweekend=md.MO))  #每周一MO
>ax.xaxis.set_major_locator(md.DateFormatter(%d %b %Y))
>ax.xaxis.set_minor_locator(md.DayLocator())
>dates = dates.astype(md.datetime.datetime)
>plt.plot(dates,prise,color='dodgerblue',label='xxx',linestyle='--',linewidth=2，alpha = 0.3)
>#蜡烛图
>#color
>rise = closing_prises>opening_prices
># color=['white'if x else 'green' for x in rise]
>color = np.zeros(rise.size,dtype='U5')
>color[:]='green'
>color[rise]='white' #mask
>#边缘色
>ecolor= ['red' if x else 'green'for x in rise]
>#影图
>plt.bar(dates,closing_prices-opening_prices,0.8,opening_prices,color=[],edgecolor=ecolor,zorder=3)
>#线图
>plt.vlines(dates,lowest_prices,highest_prices,color=ecolor)
>
>plt.legend()
>plt.gcf()autofmt_xdate()
>plt.show()
>```

#### 2. 算数平均值 nean

>真值的无偏估计
>
>```python
>m = (s1+s2+s3+...+sn)/n
>np.mean(array)
>array.mean()
>```
>
>```python
>mean = 0
>for closing_price in closing_prices:
>    mean+=closing_prise
>mean/=closing_price.size
>mean = np.mean(closing_prices)
>```
>
>```python
>def dmy2ymd(dmy):
>    dmy= str(dmy,encoding='utf-8')
>    time = dt.datetime.steptime(dmy,'%d-%m-%Y').date()
>    t = time.strftime('%Y=%m=%d')
>    return t
>plt.figure('xxx',facecolor='lightgray')
>plt.title('xxx',fontsize=16)
>plt.xlabel('date',fontsize=14)
>plt.ylabel('prise',fontsize=13)
>plt.grid(linstyle=':')
>import matplotlib.dates as md
>ax = plt.gca()
>ax.xaxis.set_major_locator(md.WeekdayLocator(byweekend=md.MO))  #每周一MO
>ax.xaxis.set_major_locator(md.DateFormatter(%d %b %Y))
>ax.xaxis.set_minor_locator(md.DayLocator())
>dates = dates.astype(md.datetime.datetime)
>plt.plot(dates,prise,color='dodgerblue',label='xxx',linestyle='--',linewidth=2)
>#mean
>mean=np.mean(closing_prices)
>plt.hlines(mean,date[0],dates[-1],color='green',label ='mean')
>plt.legend()
>plt.gcf()autofmt_xdate()
>plt.show()
>
>```
>
>

#### 3. 加权平均值

>a = (s1w1+s2w2+…+snwn)/(w1+w2+…+wn) 
>
>VWAP:成交量加权平均价格
>
>TWAP:时间加权平均价格
>
>```python
>np.average(closing_prices,weights=volumes) #维度相同
>```
>

#### 3. 最值

>```python
>np.max()
>np.min()
>np.ptp()#极差
>a = np.random.randint(10,100,9)
>```
>
>```python
>np.argmax() #返回最大元素的下标
>plt.argmin()
>np.maximum()#两个同维6数组中对应元素最大元素构成新数组
>np.minimum()
>```
>
>```python
>min_val= np.min(lowest_prices)
>max_val= np.max(highest_prices)
>print(min_val,'~',max_val)
>min_ind= np.argmin(lowest_prices)
>max_ind= np.argmax(highest_prices)
>print('min',dates[min_ind])
>print('max',dates[max_ind])
>a= np.arange(1,10).reshape(3,3)
>b= np.arange(1,10)[::-1].reshape(3,3)
>print(np.maximum(a,b))
>```

#### 4. 中位数

>```python
>median= np.median(closing_prices)
>plt.hline(median,dates[0],dates[-1],color='violet',label='median')
>```

#### 5. 标准差

>```python
>mean = np.mean(closing_prices)#算术平均值
>devs = closing_prices-mean#离差
>dsqs = devs**2#离方差
>pvar = np.sum(dsqs)/dsqs.size#总体方差
>pstd = np.sqrt(pvar)#总体标准差
>svar = np.sum(dsqs)/(dsqs.size-1)#样本方差
>sstd = np.sqrt(svar)#样本标准差
>pstd = np.std(closing_prices)#样本方差
>sstd = np.std(closing_prices,ddof=1)#样本标准差
>```

#### 6. 时间数据处理 数据轴向汇总

>
>
>```python
>def fun(data):
>    pass
>#axis [0,1]
>np.apply_along_axis(func.axis.array)
>```

#### 7. 移动均线

收盘价五日均线：第五天开始，每天计算最近（第1-5）五天收盘价的平均值

>
>
>```python
>ma5 = np.zeros(closing_prices.size-4)
>for i in range(ma5.size):
>    ma5[i] = closing_prices[i:i+5].mean()
>```

#### 8. 卷积convolve  46

valid 有效卷积 same 同维卷积 full 完全卷积

>```python
>a = [1 2 3 4 5]#源数组
>b = [8 7 6]#卷积核数组
>    8 
>0 0 1 2 3 4 5 # pading 对应位置相乘在相加 0*6+0*7+1*8 =8
>6 7 8 
>
># 移动计算
># 卷积核数组和0乘 无效 和不是0乘生效
>                 44 65 86          #有效卷积 卷积核数组都生效
>              23 44 65 87 69       #同维卷积 结果和源数组同维
>           8  23 44 65 86 59 30    #完全卷积full 
>0  0  1  2  3  4  5  0  0     # 0*6+1*7+2*8 =23
>6  7  8 
>   6  7  8 
>      6  7  8
>         6  7  8 
>            6  7  8
>               6  7  8
>                  6  7  8
>c = numpy.convolve(a,b,卷积类型)
> 
>```
>
>```python
>#五日移动均线
>a = [a,b,c,d,e,f,g,h,i,j]#源数组
>b = [1/5,1/5,1/5,1/5,1/5]#卷积核数组
>|
>v
>kernel = np.ones(5)/5
>sma5 = np.convolve(closing_prices, np.ones(5)/5,'valid')
>```

#### 9. 加权卷积

>```python
>a = [a,b,c,d,e,f,g,h,i,j]
>b = [1/5,3/25,5/25,7/25,9/25]#权重 找公式
>```
>
>```python
>x= np.linespace(-1,0,5)
>kernel = np.exp(x)[::-1] #e的x次方  比例不变元素之和为1
>kernel= kernel/kernel.sum()
>ma5 = np.convolve(closing_prices,kernek,'valid')
>```

#### 10. 布林带 Bollinger Band

三条线组成：中轨移动平均线

​                     上轨uper 中轨+2x5日收盘价标准差

​                     下轨down 中轨-2x5日收盘标准价

收窄表示稳定趋势，张开表示比较大波动空间的趋势

>```python
>#中轨
>x = np.linespace(-1,0,5)
>kernel = np.exp(x)[::-1]
>kernel = kernel/ kernel.sum()
>ma5 = np.convelve(closing_prices,kernel,'valid')
>plt.plot(dates[4:0],ma5,color='orangerred',lavel'ma-5)
>#上轨 % 下轨
>stds = np.zeros(ma5.size)
>for i in range(stds.size):
>    stds[i]= closing_prices[i:i+5].std()
>upper = ma5 + 2*stds
>lower = ma5 - 2*stds
>plt.plot(dates[4:],upper,color='red',label='upper')
>plt.plot(dates[4:],lower,color='red',label='lower') 
>plt.fill_between(
>dates[4:0],upper,lower,lower<upper,color='orangered',alpha=0.2)
>plt.legend()
>plt.gcf().autofmt_xdate()
>plt.show()         
>```
>

#### 11. 线性预测

>```python
>'''
>a b c d e f ?
>a b c   x   d
>b c d * y = e
>c d e   z   f
>A     X   B
>
>ax+by+cz=d
>bx+cy+dz=e
>cx+dy+ez=f
>'''
>x = np.linalg.lstsq(A,B)[0]
>```
>
>```python
>#预测
>def dmy2ymd(dmy):
>dmy= str(dmy,encoding='utf-8')
>time = dt.datetime.steptime(dmy,'%d-%m-%Y').date()
>t = time.strftime('%Y=%m=%d')
>return t
>plt.figure('xxx',facecolor='lightgray')
>plt.title('xxx',fontsize=16)
>plt.xlabel('date',fontsize=14)
>plt.ylabel('prise',fontsize=13)
>plt.grid(linstyle=':')
>import matplotlib.dates as md
>ax = plt.gca()
>ax.xaxis.set_major_locator(md.WeekdayLocator(byweekend=md.MO))  #每周一MO
>ax.xaxis.set_major_locator(md.DateFormatter(%d %b %Y))
>ax.xaxis.set_minor_locator(md.DayLocator())
>dates = dates.astype(md.datetime.datetime)
>plt.plot(dates,prise,color='dodgerblue',label='xxx',linestyle='--',linewidth=2)
>#线性预测
># A B 整理
>A=np.zeros((3,3))
>for j in range(3):
>a[j,] = closing_prices[j:j+3]
>B = closing_prices[3,6]
>X = np.linalg.lstsq(A,B)[0]
>B.dot(X)
># 升级版
>pred_prices = np.zeros(30-6)
>for i in range(30-6):
>A=np.zeros((3,3))
>for j in range(3):
>a[j,] = closing_prices[i+j:i+j+3]
>B = closing_prices[i+3,i+6]
>X = np.linalg.lstsq(A,B)[0]
>pred = B.dot(X)
>pred_prices[i] = pred
>plt.plot(dates[6:],pred_prices,'o-',color='orangered',label='prediction')
># 再次升级版
>N = 3
>pred_prices = np.zeros(30-N*2)
>for i in range(30-6):
>A=np.zeros((N,N))
>for j in range(N):
>a[j,] = closing_prices[i+j:i+j+N]
>B = closing_prices[i+N,i+N*2]
>X = np.linalg.lstsq(A,B)[0]
>pred = B.dot(X)
>pred_prices[i] = pred
>plt.plot(dates[N*2:],pred_prices,'o-',color='orangered',label='prediction')
>
>plt.legend()
>plt.gcf()autofmt_xdate()
>plt.show()
>```
>

#### 12. 线性拟合

>
>
>```python
>def dmy2ymd(dmy):
>dmy= str(dmy,encoding='utf-8')
>time = dt.datetime.steptime(dmy,'%d-%m-%Y').date()
>t = time.strftime('%Y=%m=%d')
>return t
>plt.figure('xxx',facecolor='lightgray')
>plt.title('xxx',fontsize=16)
>plt.xlabel('date',fontsize=14)
>plt.ylabel('prise',fontsize=13)
>plt.grid(linstyle=':')
>import matplotlib.dates as md
>ax = plt.gca()
>ax.xaxis.set_major_locator(md.WeekdayLocator(byweekend=md.MO))  #每周一MO
>ax.xaxis.set_major_locator(md.DateFormatter(%d %b %Y))
>ax.xaxis.set_minor_locator(md.DayLocator())
>dates = dates.astype(md.datetime.datetime)
>plt.plot(dates,prise,color='dodgerblue',label='xxx',linestyle='--',linewidth=2)
>#趋势线
>#每一天趋势价格
>dates,opening_prices,highest_prices,\
>lowest_prices
>trend_prices = (highest_prices+lowest_prices+closing_prices)/3
>plt.scatter(dates,trend_prices,marker='-o',color='orangered',s=80,label='trend points')
>#绘制趋势线 整理A B 
>days = dates.astype('M8[D]'),axtype('int32')
>A = np.colum_stack((days,np.ones_like(days)))
>B = trend_prices
>x = np.linelg.lstsq(A,B)[0]
>trend_line = x[0]*days+x[1]
>plt.plot(dates,trend_line,color+'orangered',label='trend   line')
>```
>

#### 13. 协方差，相关矩阵，相关系数

>
>
>```python
>cov_ab = ave(dev_a x dev_b)
>cov_ba = ave(dev_b x dev_a)
>```
>
>```python
>a = np.random.randint(1,30,10)
>b = np.random.randint(1,30,10)
>
>ave_a = np.mean(a)
>ave_b = np.mean(b)
>#离差
>dev_a = a - ave_a
>dev_b = b - ave_b
>#协方差  接近1 正相关 -1 负相关 
>cov_ab = np.mean(dev_a * dev_b)
>cov_ba = np.mean(dev_b * dev_a)
>#相关系数
>cov_ab/(std_a * std_b)
>cov_ba/(std_b * std_a)
>#相关矩阵
>numpy.cov(a,b)
>numpy.corrcoef(a,b)
>```

#### 14. 多项式拟合

>```python
>'''
>X 自变量
>Y 实际函数值
>Y' 拟合函数值
>P  多项式函数中的系数
>'''
>np.polyfit(X,Y,最高次幂)--> p
>np.polyval(P,X)--> Y'
>np.polyder(P)--> Q # 导函数的系数
>xs = np.roots(Q) #与x轴交点的横坐标
>Q = np.polysub(P1,P2)# 两个曲线的交点
>```
>
>```python
>x = np.linespace(-20,20,1000)
>y = 4*x**3+3*x**2-1000*x+1
>p = [4,3,-1000,1]
>y = polyval(P,x)
>Q = np.polyder([4,3,-1000,1])
>xs = np.root(Q)
>ys =  4*xs**3 +3*xs**2-1000*xs+1
>ply.plot(x,y)
>plt.scatter(xs,ys,s=80,c='oranagered')
>plt.show()
>```
>
>

#### 15. 数据平滑

>卷积降噪
>
>```python
>convolve_core = np.hanning(8)
>convolve_core/= convolve_core.sum()
>bhp_returns_convolved = np.convolve(bhp_returns,convolve_core,'valid')
>vale_returns_convolved = np.convolve(vale_returns,convolve_core,'valid')
>```
>
>拟合
>
>```python
>bhp_p = np.polyfit(dates[7:],bhp_returns_convolved,3)
>bhp_polyfit_y = np.polyval(bhp_p,dates[7:])
>vale_p = np.polyfit(dates[7:],vale_returns_convolved,3)
>vale_polyfit_y = np.polyval(vale_p,dates[7:])
>
>```

#### 16. 符号数组

>```python
>ary = np.sign(源数组)
># 正数变1 复数变-1
>```
>
>OBV
>
>```python
># 数组处理函数
>ary = np.piecewise(源数组，条件序列，取值序列)
>a = np.array([70,80,60,30,40])
>d = np.piecewise(
>a,
>[a<60,a==60,a>60],
>[-1,0,1])
># d = [1 1 0 -1 -1]
>```
>
>

#### 17矢量化 vectorize

>
>```python
>def foo(x,y):
>    return m.sqrt(x**2+y**2)
>x,y = 3,4
>x,y = np.array([3,4,5]),np.array([4,5,6])
>foo_vec = np.vectorize(foo)
>print(foo_vec(x,y))
>print(np.vectorize(foo)(x,y))
>```
>
>```python
>def profit(opening_prices,highest_prices,lowest_prices,closing_price):
>    buying_prices = opening_price*0.00
>    if lowest_price<=buying_price<=highest_prices:
>        return(closing_prices-buying_price)*100/buying_price
>    return np.nan #无效值
>```
>

#### 18. 矩阵 Matrix

>```python
>numpy.matrix(
>ary,
>copy=True #是否复制矩阵 各自独立数据 不共享
>)
>```
>
>```python
>numpy.mat(任何可以被解释的二维容器) #等价 numpy.matrix(,copy=False)
>#符合拼块规则'1 2 3;4 5 6'
>```

##### 乘法运算

>
>```python
>e = np.mat('123;123;123')
>print(e*e)

##### 逆矩阵

AB=BA=E(单位矩阵)

>```python
>#单位矩阵 对角线为1 其他为0 相当于四则运算里的1
>e = np.mat('123;123;123')
>print(e.I)
>print(e*e.I)
>x = np.linalg.solve(a,b)#解方程
>```

#### 19. 通用函数

>```python
>#数组剪裁 低于最小值变成最小值 最大同理
>ndarray.clip(ary，min=下限，max=上限)
>
>#数组压缩  返回满足条件的新数组
>ndarray.compress(条件ary>10)
>```
>
>加法乘法
>
>```python
>add(a,b)#数组相加
>add.reduce(a)#数组累加和
>add.accumulate(a)#累加过程
>add.outer([1,2,3],a)#外和
>ndarray.prod()#返回调用数组中所有元素的乘积/累乘
>ndarray.cumprod()#返回调用数组的元素执行累乘的过程数组
>np.outer([1,2,3],a)#外积 ([1*a]
>                 #  [2*a]
>                 #  [3*a])
>```
>
>除法
>
>```python
>np.divide(a,b)#真除
>np.floor(a/b)#地板除 真除的结果向下取整
>np.ceil(a/b)#天花板除 真除的结果向上取整
>np.trunc(a/b)#截断除 真除的结果去掉小数部分 truncate
>np.round(a/b)#四舍五入取整
>```
>
>位运算
>
>```python
>c = a^b #异或  是否同号+ -
>c = a.__xor__(b)
>c = np.bitwise_xor(a,b)
>print(np.where(c<0))
>```
>
>```python
>#1 2^0 00001 0 00000
>#2 2^1 00010 1 00001
>c = a&b #与  是否2的幂 n&(n-1)==0
>c = a.__and__(b)
>c = np.bitwise_and(a,b)
>```
>
>```python
>__or__
>bitwise_or
>```
>
>```python
>~ #位反
>__not__
>bitwise_not
>~0 = 1
>```
>
>```python
>#移位 相当于乘除2
><<  __lshift  left_shift 
>>>  __rshift  right_shift
>```
>
>三角函数
>
>```python
>numpy.sin() 
>```
>
>```python
>x=np.linespace(0,np.pi*4,1000)
>y1=4*np.pi*np.sin(x)
>'''
>傅里叶 波叠加
>y = np.zeros(n)
>for i in range(1,n+1)
>	y += 4/(2*i-1)*np.pi*\
>	np.sin((2*i-1)*x)
>'''
>plt.grid(linestyle=':')
>plt.plot(x,yi)
>plt.legend()
>plt.show()
>```
>
>

#### 20. 特征值&特征向量

>
>```python
>'''
>已知n 阶方阵A 求特征值和特征向量
>eigvals 特征值数组
>evigvecs 特征向量数组
>diag(eigvals) 特征向量对角阵 对角线上有元素 其他0
>'''
>#提取特征值
>eigvals,eigvecs = np.linalg.eig(A)
>#已知特征值&特征向量 求方阵
>#S = eigvecs * diag(eigvals) * eigvecs逆
>S = np.mat(eigvecs)*np.mat(np.diag(eigvals))*np.mat(eigvecs逆)
>
>```

>
>
>```python
>#读取图片 提取特征值和向量 保留部分特征值 重新生成矩阵 绘制图片
>
>import numpy as np
>import scipy.misc as sm
>import matplotlib.pyplot as plt
>
>original = sm.imread('',true) #true 提取亮度矩阵 rgb 变成一维
>
>#提取特征值
>original = np.mat(original)
>eigvals,eigcecs = np.linalg.eig(original)
>#抹去信息
>eigvals[50:] = 0
>#dst = np.mat(eigvecs)*np.mat(np.diag(eigvals))*np.mat(eigvecs).I
>dst = eigvecs*np.diag(eigvals)*eigvecs.I
>
>plt.subplot(121)
>plt.imshow(original,cmap='gray')
>plt.xticks([])
>plt.yticks([])
>plt.tight_layout()
>plt.show()
>
>plt.subplot(122)
>plt.imshow(dst.real,cmap='gray') #得出的结果有复数 加.real保留实部
>plt.xticks([])
>plt.yticks([])
>plt.tight_layout()
>plt.show()
>```
>

#### 21. 奇异值分解

>矩阵M 分解成三个矩阵U,S,V(U*S*V=M) U&V是正交阵（乘以自身转置矩阵结果为单位矩阵）
>S矩阵主对角线上的值称为M奇异值 其他均为0
>
>```python
>M = np.lmat('1 11 14; 8 7 -2')
>U,sv,V = np.linalg.svd(M,full_matrices=False)
>U*U.T
>V*V.T
>S = np.diag(sv)
>U*S*V
>```
>
>```python
>#读取图片 提取特征值和向量 保留部分特征值 重新生成矩阵 绘制图片
>
>import numpy as np
>import scipy.misc as sm
>import matplotlib.pyplot as plt
>
>original = sm.imread('',true) #true 提取亮度矩阵 rgb 变成一维
>
>#提取特征值
>original = np.mat(original)
>eigvals,eigcecs = np.linalg.eig(original)
>#抹去信息
>eigvals[50:] = 0
>#dst = np.mat(eigvecs)*np.mat(np.diag(eigvals))*np.mat(eigvecs).I
>dst = eigvecs*np.diag(eigvals)*eigvecs.I
>
>#奇异值分解
>originnal = np.mat(original)
>U,sv,V= np.linalg.svd(original)
>sv[50:] = 0
>dst2 = U*np.diag(sv)*V
>
>plt.subplot(221)
>plt.imshow(original,cmap='gray')
>plt.xticks([])
>plt.yticks([])
>plt.tight_layout()
>
>plt.subplot(222)
>plt.imshow(dst2.real,cmap='gray') #奇异值分解图
>plt.xticks([])
>plt.yticks([])
>plt.tight_layout()
>
>plt.subplot(224)
>plt.imshow(dst.real,cmap='gray') #得出的结果有复数 加.real保留实部
>plt.xticks([])
>plt.yticks([])
>plt.tight_layout()
>plt.show()
>```

#### 22. 快速傅里叶变换（fft）

>```python
>import numpy.fft as nf
>
>freqs = np.fftfreq(采样数量，采样周期)
>
>numpy.fft.fft(原函数值序列)->目标函数值序列（复数）
>
>numpy.fft.ifft(目标函数值序列)->原函数值序列
>```
>
>```python
>x=np.linespace(0,np.pi*4,1000)
>y1=4*np.pi*np.sin(x)
>
>#傅里叶 波叠加
>y = np.zeros(n)
>for i in range(1,n+1)
>	y += 4/(2*i-1)*np.pi*\
>	np.sin((2*i-1)*x)
>
># 波拆解 合成
>complex_ary = nf.fft(y)    
>y_= nf.ifft(complex_ary)
>plt.plot(x,y_,label='y_',color='red',linewidth=7,alpha=0.2)
>
>#频域图像  频率/能量图像
>freqs = nf.fftfreq(y_.size,x[1]-x[0])
>pows = np.abs(complex_ary) #复数的模 能量值
>
>plt.grid(linestyle=':')
>plt.plot(x,yi)
>plt.legend()
>plt.show()
>```

#### 23. 傅里叶降噪 音频识别 MSCC

>```python
>import numpy as np
>import numpy.fft as nf
>import scipy.io.wavfile as wf
>import matplotlib.pylot as plt
>
>```

#### 24. 随机数模块(Random)

>```python
>#binomial 二项分布
>#产生size个随机数  来自n次尝试成功次数 概率p
>np.random.binomial(n,p,size)
>sum(np.random.binomial(10,0.3,200000)==5)/200000#投篮命中率0.3 投10次中5次的概率
>
>#超几何分布 hypergeometric
>#产生size个 随机数为总样本抽nsample个样本后好样本的个数，总样本=ngood好样本+nbad坏样本
>np.random.hypergeometric(ngood,nbad,nsample,size)
>#7个好苹果 3个坏 抽三个好 概率
>np.random.hypergeometric(7,3,3,100000)
>#正态分布
>np.random.normal(size)
>```

#### 25. 杂

##### 排序

>```python
>np.msort()
>
>#联合间接排序
>numpy.lexsort((次排序序列，主排序序列))#依照两个特征，先年龄后性别
>
>names = np.array(['a','b','c','d'])
>prices = np.array([100,200,300,400])
>volume = np.array([100,80,50,20])
>indices = np.lexsort((-volumes,prices))
>names[indices ]
>
>#复数数组排序
>numpy.sort_complex
>
>#插入排序
>numpy.searchsorted(有序序列，待插数列)# 插入依旧有序  返回插入索引
>numpy.insert(被插序列，位置序列，待插数列) #返回插入后结果
>```

##### 插值

>```python
>import scipy.interpolate as si
>func = si.interp1d(
>离散水平坐标，
>离散垂直坐标，
>kind=插值算法(缺省为线性插值)
>)
>```
>
>```python
>import scipy.interpolate as si
>
>min_x =-50 
>max_x = 50
>x = np.linspace(min_x,max_50,15)
>y = np.sinc(x)
>
>plt.grid(linestyle=':')
>plt.scatter(x,y,s=60,color='dodgerblue',marker='o',label='samples')
>
>func = si.interp1d(x，y，kind='linear')
>linear_x = np.linspace(min_x,max_x,1000)
>linear_y = linear(linear_x)
>plt.plot(linear_x,linear_y,color='green',label='linear interpolation')
>#三次样条插值器
>cubic = si.interp1d(x，y，kind='cubic')
>cubic_x = np.linspace(min_x,max_x,1000)
>cubic_y = linear(linear_x)
>plt.plot(cubic_x,cubic_y,color='red',label='linear interpolation')
>
>plt.legend()
>plt.show
>```
>
>

##### 积分

>```python
>#函数f 积分上下限a,b  返回(积分值，最大误差)
>area = scipy.integrate.quad(f,a,b)
>
>```

##### 图像

>```python
>import scipy.ndimage as sn
>import scipy.misc as sm
>
>original = sm.imread('',Ture)
>#高斯模糊
>median = sn.median_filter(original,21)
>#角度旋转
>rotate = sn.rotate(original,45)
>#边缘识别
>prewitt = sn.prewitt(original)
>```

##### 金融

>
>
>```python
>np.fv(利率，期数，每期支付，现值)
>np.fv(0.01，5，-100，-1000)#1000元 1%利率存银行 存5年 每年加1000元 最后本息
>
>np.pv(利率，期数，每期支付，终值)
>np.pv(0.01，5，-100，fv)
>#多少钱 1%利率存银行 存5年 每年加1000元 最后本息fv元
>
>净现值=np.ppv(利率，现金流)
>#1000 1%利息存5年 每年加100 相当于一次存多少
>np.ppv(0.01，[-1000,-100,-100,-100,-100,-100])
>
>#每期支付
>np.pmt(利率，期数，现值)
>
>#期数
>np.nper(利率，每期支付，现值)
>
>#利率
>np.rate(期数，每期支付，现值，终值)
>```
>
>

### Pandas

---

#### 核心数据结构

>Series 类似数组 有序字典 index value
>
>DataFrame 由多个Series组成
>
>创建Series
>
>```python
>import pandas as pd
>import numpy as np
>#创建Series
>
>#空的
>s = pd.Series()
>#ndarray
>data = np.array(['a','b','c','d'])
>s = pd.Series(data)
>s = pd.Series(data,index=['100号','101','102号','103号'])
>print(s['103号'])
>#字典
>data = {'az':22,'bz':144,'cc':25,'ad':445}
>s = pd.Series(data)
>print(s['az'])
>#标量
>s = pd.Series(5,index=[0,1,2,3])
>s = pd.Series(1/5,index=np.arange(5))#5 个1/5
>
>print(s,type(s),s.dtype)
>```
>
>访问Series
>
>```python
>#索引 & 标签
>data = np.array(['a','b','c','d'])
>s = pd.Series(data)
>s[0] & s['a']
>```

#### 日期

>Series.dt
>
>```python
>#识别日期格式
>dates = pd.Series(['2021','2011-02','2011-02-22','2011/03/11','2011/03/11 01:01:01','o1 Jun 2022'])
>#转换类型
>dates = pd.to_datetime(dates)
>print(dates,dates.dtype,type(dates))
>#日期运算
>delta = dates - pd.to_datetime('1970-03-11')
>#天数转数字
>pritn(delta.dt.days)
>```
>
>DateTimeIndex
>
>时间当作索引pd.date_range()
>
>```python
>#日为频率
>datelist = pd.date_range('2019/08/21',period=5) #向后偏移5天
>#月为频率
>datelist = pd.date_range('2019/08/21',period=5,preq='M')#向后偏移5月
>#区间的时间序列
>start = pd.datetime(2017,11,1)
>end = pd.datetime(2017,11,5)
>dates = pd.date_range(start,end)
>#工作日序列
>pd.bdate_range('2019/08/01',period=22) 
>```
>
>

#### DataFrame

二维数组 行列两个索引

>创建DF
>
>```python
>#空df
>df = pd.DataFrame()
>#列表
>data = [1,2,3,4,5]
>df = pd.DataFrame(data)
>
>data = [['zdz',151],['zgz',141],['hzz',113]]
>df = pd.DataFrame(data,columns=['name','scores'])
>
>data = [['zdz',151],['zgz',141],['hzz',113]]
>df = pd.DataFrame(data,columns=['name','scores'],dtype=float)
>
>data = [{'zdz':11,'zgz':11},{'zdz':151,'zgz':141,'hzz':113}]
>df = pd.DataFrame(data,columns=['name','scores'],dtype=float)
>#字典
>data = {'name':['zdz','zgz','hzz'],'scores':[151,141,113]}
>data = pd.DataFrame(data,index=['a','b','c','d'])
>
>data = {'one':pd.Series([1,2,3],index=['a','b','c']),
>       'two':pd.Series([1,2,3,4],index=['a','b','c','d'])
>       }
>df = pd.DataFrame(data)
>
>```
>

#### 核心数据结构操作

>列 相当于series
>
>列添加
>
>
>```python
>data = {'name':['zdz','zgz','hzz'],'scores':[151,141,113]}
>data = pd.DataFrame(data,index=['a','b','c'])
>df['age'] = pd.Serise([11,22,33],index=['a','b','c'])
>df['gender'] = pd.Serise([1,2,2],index=df.index)
>```
>
>列删除
>
>```python
>del(df['name'])
>
>df.pop('age')
>```
>
>行
>
>访问
>
>```python
>df[2:4]
>#loc 方法 接受 index
>df.loc[['zdz','hzz']]
>df.loc['zdz']
># iloc 方法接受 下标索引
>df.iloc[[1,2]]
>```
>
>添加
>
>```python
>df.append(df)
>
>data1 = [['zdz',151],['zgz',141],['hzz',113]]
>df1 = pd.DataFrame(data1,columns=['name','scores'],dtype=float)
>data2 = [['zdz',151],['zgz',141],['hzz',113]]
>df2 = pd.DataFrame(data2,columns=['name','scores'],dtype=float)
>df = df1.append(df2)
>```
>
>删除
>
>```python
>df = df.drop(0) #变量接受结果
>```
>
>修改
>
>```python
>df['name'][0] = 'kaksj'
>df.iloc[0]['aga'] = 'ljk'
>```

#### 属性

>```python
>axes 返回行列的index列表
>dtype 
>empty  如果空 但会 true 
>adim   数据维度 默认：1
>size   元素个数
>values 将系列作为ndarray返回
>head() 返回前n行
>tail() 返回最后n行
>```
>
>

#### 描述description

>```python
>np.min
>np.max
>np.mean
>np.ptp#极差
>np.median
>np.std#标准差
>np.var#方差
>np.cov#协方差
>
>pd.sum()
>pd.sum(1)
>pd.mean()
>pd.mean(1) #1 水平方向
>
>sum()
>count() #非空观测数量
>mean()
>meidan
>std()
>min()
>max()
>abs()
>prod()
>cumsun()#累计综合
>cumprod()#累计乘积
>describe(includ=['objece','number'])#非空数目，均值，标准差，四分位
>```

#### 排序

>按照标签sort_index()  传axis和排序顺序
>
>
>```python
>unsorted_df = pd.DataFrame(np.random.randn(10,2),
>                          index=[1,4,6,2,3,5,9,8,0,7],columns=['col2','col1'])
>#按照标签
>sorted_df = unsorted_df.sort_index()
>sorted_df = unsorted_df.sort_index(ascending=False)
>#axis
>sorted_df = unsorted_df.sort_index(axis=1)#水平
>```
>
>按某列 sort_values()接受by 参数
>
>```python
>sorted_df = unsorted_df.sort_valurs(by='age',ascending=False)
>#先年龄拍再分数
>sorted_df = unsorted_df.sort_valurs(by=['age','scores'],ascending=[False,True])
>```
>

#### 分组group by

>```python
>#按照year分组
>grouped = df.groupby('year')
>#查看
>grouped.groups
>#迭代
>for year,group in grouped:
>    print(year)
>    print(group)
>#细节
>group_2014 = grouped.get_group(2014)
>
>```
>
>分组聚合 agg()
>
>```python
>#每一年平均的分
>grouped = df.groupby('year')
>grouped['points'].agg(np.mean) #agg 聚合
>
>grouped = df.groupby('year')
>grouped['points'].agg([np.mean,np.sum,np.std])
>```

#### 透视表交叉表 cross-tabulation

>```python
>data.pivot_table(index=['class_id','gender'])
>data.pivot_table(index=['class_id','gender'],columns=['age'],values=['score'],aggfunc='max')
>```
>
>



#### 数据表关联 merge()

>```pythonda
>left = pd.DataFrame()
>right = pd.DataFrame()
>rs = pd.merge(left,right)
>```
>
>
