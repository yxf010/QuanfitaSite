from django import forms
from captcha.fields import CaptchaField
from .models import UserProfile as User


class UserForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=128, widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(label="密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    captcha = CaptchaField(label='验证码')

class RegisterForm(forms.Form):
    gender = (
        ('male', "男"),
        ('female', "女"),
    )
    username = forms.CharField(label=u"用户名", max_length=128, widget=forms.TextInput(attrs={'class': 'form-control'}))
    password1 = forms.CharField(label=u"密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    password2 = forms.CharField(label=u"确认密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(label=u"邮箱地址", widget=forms.EmailInput(attrs={'class': 'form-control'}))
    sex = forms.ChoiceField(label=u'性别', choices=gender)
    captcha = CaptchaField(label=u'验证码')

def upload_to(instance, filename):
    return '/'.join(['user', instance.username, filename])

class InfoForm(forms.Form):
    """docstring for InfoForm"""
    
    gender = (
        ('male', "男"),
        ('female', "女"),
    )
    
    avatar = forms.FileField(
        #upload_to=upload_to,
        #default="media/user/default.svg",
        #max_length=500,
        allow_empty_file=True,
        label=u"头像"
    )
    username = forms.CharField(label=u"用户名", max_length=128, widget=forms.TextInput(attrs={'class': 'form-control'}))
    nick_name = forms.CharField(label=u"昵称", max_length=128, widget=forms.TextInput(attrs={'class': 'form-control'}))
    #password1 = forms.CharField(label="密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    #password2 = forms.CharField(label="确认密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(label=u"邮箱地址", widget=forms.EmailInput(attrs={'class': 'form-control'}))
    mobile = forms.RegexField('^((13[0-9])|(14[5,7])|(15[0-3,5-9])|(17[0,3,5-8])|(18[0-9])|166|198|199|(147))\\d{8}$',
        max_length=11, min_length=11, required=False, label=u"手机号", 
        widget=forms.TextInput(attrs={'class': 'form-control'}))
    sex = forms.ChoiceField(label=u'性别', choices=gender)
    birthday = forms.DateField(label=u"生日",required=False, widget=forms.DateInput(attrs={'class': 'form-control'}))
    #captcha = CaptchaField(label='验证码')
    signature=forms.CharField(max_length=200,required=False, label=u"个性签名", widget=forms.Textarea(attrs={'class': 'form-control'}))
    