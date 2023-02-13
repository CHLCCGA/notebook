# day20 Django开发和爬虫

day 19 [源码](https://github.com/qingDragon/django_day19)

[用户管理系统](https://gitee.com/dongxiaochuan/sms-b)

[源码 系统设计](https://blog.csdn.net/Tsuki_0/article/details/119884210)

[源码 网页部署](https://blog.csdn.net/Tsuki_0/article/details/119796360)

## 1.工单系统

- 时间问题：相差8小时

  ```python
  TIME_ZONE = 'UTC'
  USE_TZ = True
  ```

  ```python
  # datetime.datetime.now() - 东八区时间 / datetime.datetime.utcnow() => utc时间
  TIME_ZONE = 'Asia/Shanghai'
  
  USE_TZ = False  # 根据TIME_ZONE设置的时区进行创建时间并写入数据库
  ```

- 错误提示（ajax请求）

  - 视图的返回值

    ```python
    def my_add_plus(request):
        """ ajax提交并创建 """
        form = MyModelForm(data=request.POST)
        if not form.is_valid():
            return JsonResponse({'status': False, 'error': form.errors})
    
        tpl_object = form.cleaned_data['tpl']
        form.instance.user_id = request.unicom_userid
        form.instance.leader_id = tpl_object.leader_id
        form.instance.create_datetime = datetime.now()
        form.save()
        return JsonResponse({'status': True})
    ```

  - 页面发起Ajax请求

    ```javascript
    $.ajax({
        url: "/my/add/plus/",
        type: "POST",
        data: data,
        dataType: "JSON",
        success: function (res) {
            if (res.status) {
                location.reload();
            } else {
                // { 'status':False,'error':{ 'tpl':["这个字段不能为空"]} }
                $.each(res.error, function (key, valueList) {
                    //console.log(key,valueList);
                    $("#id_" + key).next().text(valueList[0]);
                })
            }
        }
    })
    ```

  - 标签和错误提示位置

    ```html
    <div class="col-sm-10" style="position: relative;">
        {{ field }} - <input id='id_tpl' ... />
        <span class='error-message' style="color: red;position: absolute;"></span>
    </div>
    ```

  - 清除错误

    ```html
    $(".error-message").empty();
    ```





### 1.1 工单审批

![image-20220802100452738](assets/image-20220802100452738.png)



### 1.2 文件上传

- 上传文件

  ```html
  <form method="post" enctype="multipart/form-data">
      {% csrf_token %}
      <input type="file" name="fs">
      <input type="submit" value="提 交">
  </form>
  ```

  ```python
  def up(request):
      if request.method == "GET":
          return render(request, 'up.html')
  	# 读取文件对象
      file_object = request.FILES.get('fs')
      print(file_object.name)
      print(file_object.size)
  
      # 读取文件中的数据，上传到服务器
      with open(file_object.name, mode='wb') as f:
          for chunk in file_object.chunks():
              f.write(chunk)
      return HttpResponse("测试")
  ```

- 上传图片 + 看 + static （一般别这么干）

  ```python
  def up(request):
      if request.method == "GET":
          return render(request, 'up.html')
      # request.GET
      # request.POST
      file_object = request.FILES.get('fs')
  
      file_path = os.path.join('app01', 'static', 'images', file_object.name)
      with open(file_path, mode='wb') as f:
          for chunk in file_object.chunks():
              f.write(chunk)
  
      return HttpResponse("测试")
  ```

  ```
  关于项目开发中的静态资源：
  	- static，开发程序所需的自己的静态文件
  	- media，用户上传的。
  ```

- 上传图片 + 看 + media（推荐）

  ```html
  <form method="post" enctype="multipart/form-data">
      {% csrf_token %}
      <input type="file" name="fs">
      <input type="submit" value="提 交">
  </form>
  ```

  ```python
  def up(request):
      if request.method == "GET":
          return render(request, 'up.html')
      # request.GET
      # request.POST
      file_object = request.FILES.get('fs')
  
      file_path = os.path.join('media', file_object.name)
      with open(file_path, mode='wb') as f:
          for chunk in file_object.chunks():
              f.write(chunk)
  
      return HttpResponse("测试")
  ```

  如果想要观看media就需要做如下配置

  - 在settings.py

    ```python
    import os
    MEDIA_ROOT = os.path.join(BASE_DIR, "media")
    MEDIA_URL = "/media/"
    ```

  - 在URL中添加路由

    ```python
    
    from django.views.static import serve
    from django.urls import re_path
    from django.conf import settings
    
    urlpatterns = [
        # path('admin/', admin.site.urls),
        path('login/', account.login, name='login'),
        ...
        
        re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}, name='media'),
    ]
    ```

  - 在咱们项目中，特殊配置：

    ```python
    UNICOM_PERMISSION = {
        'leader': {'up','media' },
        'user': {"my_add_plus",'media'}
    }
    ```

- ModelForm实现自动上传

  ```python
  class Info(models.Model):
      name = models.CharField(verbose_name="姓名", max_length=32)
      avatar = models.FileField(verbose_name="头像", upload_to='xxxx/')
  
  ```

  ```python
  class UpModelForm(forms.ModelForm):
      class Meta:
          model = models.Info
          fields = "__all__"
  
  
  def up_form(request):
      if request.method == "GET":
          form = UpModelForm()
          return render(request, 'up_form.html', {'form': form})
  
      form = UpModelForm(data=request.POST, files=request.FILES)
      if not form.is_valid():
          return render(request, 'up_form.html', {'form': form})
  
      form.save()
      return HttpResponse("成功")
  ```

  ```html
  <form method="post" enctype="multipart/form-data">
      {% csrf_token %}
      {% for field in form %}
          {{ field }}
      {% endfor %}
      <input type="submit" value="提 交">
  </form>
  ```

  

### 1.3 Excel上传+读取

```python
def up_excel(request):
    if request.method == "GET":
        return render(request, 'up_excel.html')

    file_object = request.FILES.get('fs')
    print(file_object.name)

    # Excel格式数据 openpyxl模块
    from openpyxl import load_workbook
    wb = load_workbook(file_object)

    sheet = wb.worksheets[0]
    for row in sheet.iter_rows():
        print(row[0].value, row[1].value)
        # models.Info.objects.create(name=row[0].value, avatar=row[1].value)

    return HttpResponse("测试")
```

```python
{% extends 'layout.html' %}


{% block content %}
    <h3>上传文件</h3>
    <form method="post" enctype="multipart/form-data">
        {% csrf_token %}
        <input type="file" name="fs">
        <input type="submit" value="提 交">
    </form>
{% endblock %}
```





### 答疑-插件定制

- 修改Model

```python
class Info(models.Model):
    name = models.TextField(verbose_name="姓名", max_length=32)
    avatar = models.FileField(verbose_name="头像", upload_to='xxxx/')
```

```python
class UpModelForm(forms.ModelForm):
    class Meta:
        model = models.Info
        fields = "__all__"
```



- 修Form插件

  ```python
  class Info(models.Model):
      name = models.CharField(verbose_name="姓名", max_length=32)
      avatar = models.FileField(verbose_name="头像", upload_to='xxxx/')
  
  ```

  ```python
  class UpModelForm(forms.ModelForm):
      class Meta:
          model = models.Info
          fields = "__all__"
          widgets = {
              'name': forms.Textarea()
          }
  ```

  ```python
  class UpModelForm(forms.ModelForm):
      name = forms.CharField(
          widget=forms.Textarea()
      )
  
      class Meta:
          model = models.Info
          fields = "__all__"
  ```



## 2.知识点回顾

### 2.1 基础入门

- 编码

  ```
  - 文件以什么编码存储，就要以什么编码打开。
  - 常见编码：ascii、unicode、utf-8、gbk   -> utf-8   -> 中文3个字节
  - Python解释器编码
  	- py2，默认解释器编码ascii
  	- py3，默认解释器编码utf-8
  ```

- 输入输出

- 变量

  ```
  - 字母、数字、下划线
  - 数字不能开头
  - 不能内置关键字：def class break continue if while else 
  ```

- for循环、while循环

  ```
  for+range -> 前取后不取
  ```

- 字符串格式化

  ```
  v1 = "我是{},年龄是{}".format("武沛齐",123)
  
  v1 = "我是{0},年龄是{1}".format("武沛齐",123)
  
  v1 = "我是{0},年龄是{0}".format("武沛齐")
  ```

- 运算符

  ```
  加减乘删
  ```

### 2.2 数据类型

- 字符串

  ```
  - 不可变
  - strip/split/replace/join
        v1 = "root"
        data = v1.upper()
        print(v1)
        print(data)
  - 索引、切片、长度、循环、包含
  	 v1 = "root"
  	 v1[0]
  	 v1[0:3]
  ```

- 列表

  ```
  - 可变
  - append/insert/remove...
  	v1 = [11,22,33]
  	res = v1.append(44)
  	print(v1)
  	print(res) # None
  - 索引、切片、长度、循环、包含
  	 v1 = [11,22,33]
  	 v1[0]
  	 v1[0:2]
  ```

- 字典

  ```
  - 可变类型
  - items/keys/values/get
  - 键是有要求：可哈希。  -> int/str/bool/元组
  - 取值
  	v1 = {'k1':123,'k2':456}
  	v1['k1']
  	v1.get('k1')
  ```

- 元组

  ```
  - 不可变类型
  - 一个元素的元组：   
  	v1 = (1)
  	v2 = (1,)
  	v3 = 1
  ```

- 集合

  ```python
  v1 = {11,22,33,44}
  ```

- 浮点型

- None

- 布尔类型

  ```
  那些转换成布尔值为False:   0、""、空字典、空元祖、空集合、None
  ```



推导式：

- 列表推到

  ```
  v1 = [ i for i in range(10)]   # [0,1,2,3..9]
  v1 = [ i for i in range(10) if i>5]   # [6,7,8,9]
  ```

- 字典推导式

  ```python
  v1 = { i:100 for i in range(5)}             # {0:100,1:100,2:100,3:100,4:100,}
  v1 = { i:i for i in range(5) }              # {0:0,1:1,2:2,3:3,4:4,}
  v1 = { i:i for i in range(5) if i>2}        # {3:3,4:4,}
  ```

  

### 2.3 函数

- 定义

  ```
  def xxx():
  	pass
  ```

- 参数

- 返回值

  ```
  - 没返回值， 默认返回None
  - 返回值
  	- 值
  	- 退出函数
  ```

- 内置函数

  ```
  hex/bin/oct/mac/min.....
  ```

- 文件操作

  ```
  - 打开模式：  r/w/a      rb/wb/ab
  - 上下文
  	with open('xxxxx') as f:
  		f....
  ```

### 2.4 模块

- 分类
  - 内置模块
  - 第三方模块
  - 自定义模块
- 导入：sys.path

- 关于内置

  ```
  time/datetime/json/hashlib/random/re/xml/configparse
  
  re模块
  	- 正则  \d  \w   + ?;     贪婪匹配，不想贪婪就是在数量后面?
  	- re.search/re.match/re.findall
  ```

- 第三方模块

  ```
  pip install 第三方模块
  第三方模块：
  	- requests
  	- bs4
  	- openpyxl
  	- python-docx
  	- flask，Web框架
  	- django，Web框架
  ```



### 2.5 面向对象

```
三大特性：继承、封装、多态
成员：
	- 实例变量、类变量
	- 实例方法、类方法、静态方法
```



### 2.6 MySQL数据库

```
- 数据库 -> 文件夹
- 数据表 -> Excel文件
- 数据行 -> 数据
```

```
show databases;
use jx;

show tables;

insert into 表...
delete from xxx 
update ..
select 
```



基于python连接MySQL：

```
pip install pymysql
```

```
pip install MySQLdb
pip install mysqlclient
```

- 记住进行

  - 更新，commit
  - 删除，commit
  - 添加，commit
  - 查询

- 防止SQL注入

  ```python
  import pymysql
  
  conn = pymysql.connect...
  cursor = conn.cursor()
  
  # 建议
  cursor.execute("select * from xx where id=%s", [11])
  
  # 注入风险
  cursor.execute("select * from xx where id=%s".format(11))
  ```

  

### 2.7 前端

- HTML

  ```
  a/div/span/img/table....
  - 跨级和行内
  	div/ul/li
  	a/span/img
  - form表单
  	<form method="get" action="提交地址" encrypt='...'></form>
  ```

- CSS

  ```
  - 选择器
  	<style>
  		#v1{}
  		.v2{}
  	</style>
  	
  - 样式
  	color:xxx
  	border:
  ```

- JavaScript & jQuery

  ```
  - 选择器
  	$("#v1")
  	$(".v2")
  	
  - 操作
  	$(".v2").text()				<div>sdfsdf</div>
  	$(".v2").text("xxxx")
  	
  	$("#v2").val()				<input type='text' id='v2' value='111' />
  	$("#v2").val(222)
  ```

- BootStrap

  ```
  - 栅格：12份
  - 组件：
  	- container
  	- 面板
  	- 表单
  	- 按钮
  	- 对话框
  ```

- 关于注释

  ```html
  <html>
      <head>
          <style>
              .xxx {
                  /* xxx */
              }
          </style>
      </head>
      <body>
          <!-- <a></a>  -->
          
          <script>
          	// var v1 = 123;
              /* var v2 = 456; */
          </script>
      </body>
  </html>
  ```

### 2.8 Django

- 安装

  ```
  >>>pip install django
  ```

- 创建项目

  ```
  >>>django-admin startproject 项目名
  ```

- 创建app

  ```
  >>>python manage.py startapp app名称
  ```

- 注册app

  ```python
  # settings.py
  INSTALLED_APP = [
      "xxx.xxx.xxx"
  ]
  ```

- 静态文件目录：static           （根目录static，已注册app的static目录找）

- 模板文件目录：templates   （根目录templates，已注册app的templates目录找）

- app/models.py

  ```python
  class UserInfo(models.Model):
      ...
  ```

  ```python
  >>>python manage.py makemigrations
  >>>python manage.py migrate
  ```

- 中间件

  ```
  process_request
  process_view
  process_response
  ```

- url.py -> 路由系统

  ```
  url -> 函数对应关系
  ```

- 视图函数

  ```python
  def func(request):
      ...
  	return 返回值
  ```

  - 请求数据

    ```
    request.GET
    request.POST
    request.FILES
    request.method
    request.path_info   -> 当前请求URL的地址 
    ```

  - 返回值

    ```
    HttpResponse     ->   JSONReponse()
    render
    redirect
    ```

- cookie和Session

  ```
  cookie，保存在浏览器
  session，在服务器端 ->  Django默认把它存储表中。
  ```

- 表单校验

  - Form：生成HTML标签、表单验证

  - ModelForm：生成HTML标签、表单验证、数据库操作

    ```
    form = ModelForm(instance=对象)
    ```

    ```python
    form = ModelForm(instance=对象,data=request.POST)
    if form.is_valid():
        form.save()
    ```

- ORM操作

  - 增

    ```
    models.User.objects.create(name='xxx',age='xxx')
    models.User.objects.create(**{"name":'xxx','age':'xxx'})
    ```

  - 删

    ```python
    models.User.objects.all().delete()
    models.User.objects.filter(id=1).delete()
    models.User.objects.filter(**{"id":1}).delete()
    ```

  - 改

    ```python
    models.User.objects.all().update(name='xxx')
    ```

  - 查

    ```python
    v1 = models.User.objects.all()
    v1 = models.User.objects.filter(id=6)
    
    v1 = models.User.objects.filter(id__gt=6)   # id >  6
    v1 = models.User.objects.filter(id__gte=6)  # id >= 6
    v1 = models.User.objects.filter(id__lt=6)   # id <  6
    v1 = models.User.objects.filter(id__lte=6)  # id <= 6
    
    v1 = models.User.objects.filter(id=6)  # id = 6
    v1 = models.User.objects.exclude(id=6)  # id != 6
    v1 = models.User.objects.exclude(age=6).filter(id=100)  # id == 6 且 age!=6
    
    Q对象，用于构造复杂的SQL条件。
    v1 = models.User.objects.filter(Q(id=1)|Q(age=10))  # id = 1 or age=10
    ```

    ```python
    # queryset = [obj,obj,obj]
    v1 = models.User.objects.all()
    ```

    ```python
    # queryset = [ {'id':1,'name':"武沛齐",'age':19}, {'id':1,'name':"武沛齐",'age':19}, ]
    v1 = models.User.objects.all().values("id",'name','age')
    ```

    ```python
    # queryset = [ (1,"武沛齐",19) ,(1,"武沛齐",19),(1,"武沛齐",19) ]
    v1 = models.User.objects.all().values_list("id",'name','age')
    ```



## 3.爬虫开发

- web爬虫，模拟浏览器发送请求获取数据。
- js逆向，网站内部集成算法，去他的网站找到算法位置，算法还原。
- app爬虫
- app逆向，找到算法位置，算法的还原。



### 3.1 web爬虫

- 汽车之家，发送请求或者HTML，提取想要的数据。

  ```
  pip install requests
  pip install BeautifulSoup4
  ```

  ```python
  import requests
  from bs4 import BeautifulSoup
  
  res = requests.get(
      url="https://www.autohome.com.cn/news/",
      headers={
          "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:102.0) Gecko/20100101 Firefox/102.0",
      }
  )
  res.encoding = 'gb2312'
  
  # 第1步：将文本交给BeautifulSoup，让他帮我们结构化处理。
  soup = BeautifulSoup(res.text, features='html.parser')
  
  # 第2步：根据特点，利用  find  findall 找到相应的标签
  part_area = soup.find(name='div', attrs={"id": "auto-channel-lazyload-article"})
  
  # 所有的li标签 = [li,li,li]
  li_list_node = part_area.find_all(name='li')
  for li_node in li_list_node:
      h3 = li_node.find(name='h3')
      if not h3:
          continue
  	
      print(h3.text)
  	
      # 找到标签，获取他的文本
      p = li_node.find(name='p')
      print(p.text)
  	
      # 找到标签，获取属性  <img src='xxx' />
      img = li_node.find(name='img')
      print(img.attrs['src'])
      print('-----------------')
  ```

- 豆瓣电影，向豆瓣发送请求，返回JSON格式。

  ```python
  import requests
  
  res = requests.get(
      url='https://movie.douban.com/j/search_subjects?type=movie&tag=热门&sort=recommend&page_limit=20&page_start=20',
      headers={
          "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
      }
  )
  
  print(res.text)
  ```

- 知乎评论

  ```python
  import requests
  
  res = requests.get(
      url='https://www.zhihu.com/api/v4/comment_v5/answers/2605273514/root_comment?order_by=score&limit=20&offset=',
      headers={
          "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
      }
  )
  
  print(res.text)
  ```

  

### 3.2 js逆向

例如：bilibili网站。

- 地址

  ```
  https://api.bilibili.com/x/click-interface/click/web/h5
  ```

- 方式

  ```
  POST
  ```

- 请求体

  ```
  aid: 686431115
  cid: 786857599
  part: 1
  lv: 0
  ftime: 1659429134
  stime: 1659429135
  type: 3
  sub_type: 0
  refer_url: 
  spmid: 333.788.0.0
  from_spmid: 
  csrf: 
  ```

  ```python
  import requests
  import re
  import json
  
  res = requests.get(url="https://www.bilibili.com/video/BV1bU4y1v7nZ")
  
  data_list = re.findall(r'__INITIAL_STATE__=(.+);\(function', res.text)
  
  data_dict = json.loads(data_list[0])
  
  aid = data_dict['aid']
  cid = data_dict['videoData']['cid']
  
  print(aid, cid)
  ```

- cookie

  ```
  buvid3  ，其他请求返回的值
  b_lsid
  _uuid
  sid
  ```

  ```
  var e = this.splitDate()
  , t = Object(f.b)(e.millisecond)
  , t = "".concat(Object(f.c)(8), "_").concat(t);
  ```

  

```python
import math
import random
import time
import uuid
import requests
import re
import json


def gen_uuid():
    uuid_sec = str(uuid.uuid4())
    time_sec = str(int(time.time() * 1000 % 1e5))
    time_sec = time_sec.rjust(5, "0")

    return "{}{}infoc".format(uuid_sec, time_sec)


def gen_b_lsid():
    data = ""
    for i in range(8):
        v1 = math.ceil(16 * random.uniform(0, 1))
        v2 = hex(v1)[2:].upper()
        data += v2
    result = data.rjust(8, "0")

    e = int(time.time() * 1000)
    t = hex(e)[2:].upper()

    b_lsid = "{}_{}".format(result, t)
    return b_lsid


video_url = "https://www.bilibili.com/video/BV1Pi4y1D7uJ"
bvid = video_url.rsplit("/")[-1]
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36"
})

res = session.get(video_url)
data_list = re.findall(r'__INITIAL_STATE__=(.+);\(function', res.text)
data_dict = json.loads(data_list[0])
aid = data_dict['aid']
cid = data_dict['videoData']['cid']

_uuid = gen_uuid()
session.cookies.set('_uuid', _uuid)

b_lsid = gen_b_lsid()
session.cookies.set('b_lsid', b_lsid)

session.cookies.set("CURRENT_FNVAL", "4048")

res = session.get("https://api.bilibili.com/x/frontend/finger/spi")
buvid4 = res.json()['data']['b_4']

session.cookies.set("CURRENT_BLACKGAP", "0")
session.cookies.set("blackside_state", "0")

res = session.get(
    url='https://api.bilibili.com/x/player/v2',
    params={
        "cid": cid,
        "aid": aid,
        "bvid": bvid,
    }
)

print(res.cookies.get_dict())
```





### 3.3 app爬虫

代码代替手机app发送请求。

- 抓包，charles
- 代码还原

```python
import requests

res = requests.post(
    url="http://cd.tibetairlines.com.cn:9100/login",
    data='grant_type=password&isLogin=true&password=123&username=alex,N'
)

print(res.text)
```



### 3.4 app逆向（主流）

有一些参数，不知道是什么？肯定是app内部对他进行了加密。

此时可以使用jadx工具对手机app进行反编译，得到的java代码。

```python
import hashlib

obj = hashlib.md5()
obj.update("123123".encode('utf-8'))

data = obj.hexdigest()
print(data)
# 4297f44b13955235245b2497399d7a93
# 4297f44b13955235245b2497399d7a93
```







## 分享

- 基础 ~模块

  ```
  https://www.bilibili.com/video/BV1m54y1r7zE
  https://gitee.com/wupeiqi/python_course
  ```

- 面向对象

  ```
  https://www.bilibili.com/video/BV18E411V7ku
  https://www.bilibili.com/video/BV1p44y157T3
  ```

- MySQL数据库

  ```
  https://www.bilibili.com/video/BV1DE411n7fU
  
  https://www.bilibili.com/video/BV15R4y1b7y9
  ```

- 前端

  ```
  课程案例
  ```

- django和实战项目

  ```
  https://www.bilibili.com/video/BV1uA411b77M
  ```

- 微信小程序

  ```
  https://www.bilibili.com/video/BV1jC4y1s7QD
  ```

- 爬虫和app逆向

  ```
  https://www.bilibili.com/video/BV1kY4y1z7RX
  ```





## 毕业项目

- PPT + 讲解





































































































