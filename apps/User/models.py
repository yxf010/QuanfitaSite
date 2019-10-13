from django.contrib.auth.models import AbstractUser
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.db import models
from django.utils import timezone as datetime
from django.utils.functional import lazy
from django.utils.translation import gettext

def upload_to(instance, filename):
    return '/'.join(['user', instance.username, filename])

class UserProfile(AbstractUser):
    # 自定义的性别选择规则
    GENDER_CHOICES = (
        ("male", u"男"),
        ("female", u"女"),
        ("none", u"保密")
    )
    # 昵称
    nick_name = models.CharField(max_length=50, verbose_name=u"昵称", default="")
    # 生日，可以为空
    birthday = models.DateField(verbose_name=u"生日", null=True, blank=True)
    # 性别 只能男或女，默认女 不能多加否则表单验证的时候会报错
    gender = models.CharField(
        max_length=7,
        verbose_name=u"性别",
        choices=GENDER_CHOICES,
        default="none")
    # 邮箱
    email = models.CharField(max_length=100, verbose_name=u"邮箱", default="")
    # 电话
    mobile = models.CharField(max_length=11, null=True, blank=True, verbose_name=u"手机号")
    # 头像 默认使用default.png,默认目录就是上传目录
    # 使用validators.FileExtensionValidator来进行限制：
    # myfile = models.FileField(upload_to="%Y/%m/%d/", validators=[validators.FileExtensionValidator(['txt', 'pdf'],message = 'myfile必须为txt,pdf格式的文件')])
    # 直接使用ImageField，就可以限制上传的文件，必须是图片，不用再使用验证器validators了，效果都是一样的
    # 如果想要使用ImageField，必须要安装Pillow库，如果没安装运行pip install Pillow安装
    avatar = models.ImageField(
        upload_to=upload_to,
        default="media/user/default.svg",
        max_length=500,
        verbose_name=u"头像"
    )
    password = models.CharField(max_length=120,verbose_name=u'密码')

    created_time = models.DateTimeField(verbose_name=u'注册时间', auto_now_add=True)
    last_signin_time = models.DateTimeField(verbose_name=u'上次登录时间', auto_now_add=True)

    signature=models.CharField(max_length=100, null=True, blank=True, verbose_name=u"个性签名", default="")

    # meta信息，即后台栏目名
    class Meta:
        verbose_name = u"用户信息"
        verbose_name_plural = verbose_name

    # 重载Unicode方法，打印实例会打印username，username为继承自abstractuser
    def __str__(self):
        return "<User: %s>" % self.username


class EmailVerifyRecord(models.Model):
    code = models.CharField(max_length=20, verbose_name=u"验证码")
    email = models.EmailField(max_length=50, verbose_name=u"邮箱")
    send_type = models.CharField(verbose_name=u"验证码类型", choices=(("register",u"注册"),("forget",u"找回密码"), ("update_email",u"修改邮箱")), max_length=30)
    send_time = models.DateTimeField(verbose_name=u"发送时间", default=datetime.now)

    class Meta:
        verbose_name = u"邮箱验证码"
        verbose_name_plural = verbose_name

    def __str__(self):
        return '{0}({1})'.format(self.code, self.email)