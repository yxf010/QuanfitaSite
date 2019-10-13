from django.utils import timezone as datetime
from django.db import models
import json
import requests

# https://www.bilibili.com/video/av68090180/
# <iframe src="//player.bilibili.com/player.html?aid=68090180&cid=118018632&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
# http://api.bilibili.com/view?type=jsonp&appkey=8e9fc618fbd41e28&id=68186226&page=1&callback=jQuery17202164493076486218_1568945060574&_=1568945060791
# Create your models here.
class Video(models.Model):
    aid = models.CharField(max_length=20, verbose_name=u"aid")
    name = models.CharField(max_length=50, verbose_name=u"视频名称")
    tags = models.CharField(max_length=10, verbose_name=u"标签")
    url = models.CharField(max_length=100, verbose_name=u"链接")
    # love_nums = models.IntegerField(default=0, verbose_name=u"点赞数")
    # fav_nums = models.IntegerField(default=0, verbose_name=u"收藏数")
    # download_nums = models.IntegerField(default=0, verbose_name=u"下载数")
    cover = models.CharField(max_length=100, verbose_name=u"封面")
    desc = models.CharField(max_length=1000, verbose_name=u"说明")
    add_time = models.DateTimeField(default=datetime.now)

    class Meta:
        verbose_name = u"视频"
        verbose_name_plural = verbose_name

    def __str__(self):
        return "<Video: %s>" % self.name

    def getdata(self,aid):
        self.aid = aid
        api_url = 'http://api.bilibili.com/view?type=json&appkey=8e9fc618fbd41e28&id=%s&page=1&callback=jQuery17202164493076486218_1568945060574&_=1568945060791' % self.aid
        self.content = json.loads(requests.get(api_url).text)
        self.name = content['title']
        self.tags = content['tag']
        self.url = "http://player.bilibili.com/player.html?aid=%s&cid=118018632&page=1" % self.aid
        self.cover = content['pic']
        self.desc = content['description']
        self.add_time = content['created_at']