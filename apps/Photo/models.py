from django.utils import timezone as datetime
from django.db import models

def upload_to(instance, filename):
    return '/'.join(['image', instance.name, filename])

# Create your models here.
class Photo(models.Model):
    name = models.CharField(max_length=50, verbose_name=u"图册名称")
    tags = models.CharField(max_length=10, verbose_name=u"标签")
    # love_nums = models.IntegerField(default=0, verbose_name=u"点赞数")
    # fav_nums = models.IntegerField(default=0, verbose_name=u"收藏数")
    # download_nums = models.IntegerField(default=0, verbose_name=u"下载数")
    image = models.ImageField(upload_to=upload_to, verbose_name=u"照片", max_length=100)
    desc = models.CharField(max_length=1000, verbose_name=u"说明")
    add_time = models.DateTimeField(default=datetime.now)

    class Meta:
        verbose_name = u"图册"
        verbose_name_plural = verbose_name

    def __str__(self):
        return "<Photo: %s>" % self.name

    def image_data(self):
        return format_html(
            '<img src="{}" width="100px"/>',
            self.image.url,
        )