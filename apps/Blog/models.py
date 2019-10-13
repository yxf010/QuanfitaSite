from django.utils import timezone as datetime
from django.db import models

class Category(models.Model):
    """分类"""
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Tag(models.Model):
    """标签"""
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Blog(models.Model):
    """文章"""
    title = models.CharField(max_length=100)
    body = models.TextField()
    author = models.CharField(max_length=20)
    created_time = models.DateTimeField(datetime.now)
    excerpt = models.CharField(max_length=200, blank=True)  # 文章摘要，可为空
    category = models.ForeignKey(Category, on_delete=True)  # ForeignKey表示1对多（多个post对应1个category）
    tags = models.ManyToManyField(Tag, blank=True)
    views = models.PositiveIntegerField(default=0)  # 阅读量

    def __str__(self):
        return "<Blog: %s>" % self.title
