from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate
from django.template.loader import get_template
from django.http import HttpResponse
from django.views import View
from django.contrib.auth.decorators import login_required
from django.contrib import auth
from django.views.decorators.csrf import csrf_exempt
from .models import UserProfile as User
from .forms import RegisterForm, UserForm, InfoForm

# Create your views here.
class LoginView(View):
    """docstring for LoginView"""
    def get(self, request):
        temp = get_template('login.html')
        tmp_type = "login"
        login_form = UserForm()
        html = temp.render(locals())
        return HttpResponse(html)

    @csrf_exempt
    def post(self, request):
        if request.session.get('is_login',None):
            return redirect('/')

        if request.method == "POST":
            login_form = UserForm(request.POST)
            message = "请检查填写的内容！"
            print(request.POST)
            print(login_form.is_valid())
            if login_form.is_valid():
                username = login_form.cleaned_data['username']
                password = login_form.cleaned_data['password']
                print(username,password)
                try:
                    user = authenticate(username=username,password=password)
                    print(user,user.is_active)
                    if user is not None and user.is_active:
                        auth.login(request, user)
                        print('login')
                        
                        request.session['is_login'] = True
                        request.session['user_id'] = user.id
                        request.session['user_name'] = user.username
                        
                        return redirect('/')
                    else:
                        message = "密码不正确！"
                except:
                    message = "用户不存在！"
            return render(request, 'login.html', locals())
     
        login_form = UserForm()
        return render(request, 'login.html', locals())

def logout(request):
    if not request.session.get('is_login', None):
        # 如果本来就未登录，也就没有登出一说
        return redirect("/")
    request.session.flush()
    auth.logout(request)
    # 或者使用下面的方法
    # del request.session['is_login']
    # del request.session['user_id']
    # del request.session['user_name']
    return redirect("/")


class RegisterView(View):
    """docstring for RegisterView"""
    def get(self, request):
        temp = get_template('login.html')
        tmp_type = "register"
        register_form = RegisterForm()
        html = temp.render(locals())
        return HttpResponse(html)

    @csrf_exempt
    def post(self, request):
        if request.session.get('is_login', None):
            # 登录状态不允许注册。你可以修改这条原则！
            return redirect("/")
        if request.method == "POST":
            register_form = RegisterForm(request.POST)
            message = "请检查填写的内容！"
            if register_form.is_valid():  # 获取数据
                username = register_form.cleaned_data['username']
                password1 = register_form.cleaned_data['password1']
                password2 = register_form.cleaned_data['password2']
                email = register_form.cleaned_data['email']
                sex = register_form.cleaned_data['sex']
                if password1 != password2:  # 判断两次密码是否相同
                    message = "两次输入的密码不同！"
                    return render(request, 'login.html', locals())
                else:
                    same_name_user = User.objects.filter(nick_name=username)
                    if same_name_user:  # 用户名唯一
                        message = '用户已经存在，请重新选择用户名！'
                        return render(request, 'login.html', locals())
                    same_email_user = User.objects.filter(email=email)
                    if same_email_user:  # 邮箱地址唯一
                        message = '该邮箱地址已被注册，请使用别的邮箱！'
                        return render(request, 'login.html', locals())
     
                    # 当一切都OK的情况下，创建新用户
     
                    new_user = User.objects.create()
                    new_user.nick_name = new_user.username = username
                    new_user.password = make_password(password1)
                    new_user.gender = sex
                    new_user.save()
                    return redirect('/login/')  # 自动跳转到登录页面
        register_form = RegisterForm()
        return render(request, 'login.html', locals())

class PersonSpaceView(View):
    """docstring for PersonSpaceView"""
    def get(self, request):
        session = request.session
        if session.get('is_login',None) and len(session.keys()) <= 3:
        	return redirect('/')
        user = User.objects.get(id=session['user_id'])
        info_form = InfoForm({'username':user.username,
            'nick_name':user.nick_name,
            'email':user.email,
            'mobile':user.mobile,
            'birthday':user.birthday,
            'signature':user.signature})
        #info_form['username'].disabled = True
        return render(request, 'personal.html', locals())

    @csrf_exempt
    def post(self, request):
        session = request.session
        user = User.objects.get(id=session['user_id'])
        if not session.get('is_login', None):
            # 登录状态不允许注册。你可以修改这条原则！
            return redirect("/")
        
        info_form = InfoForm(request.POST or None,request.FILES or None)
        print(info_form.is_valid())
        try:
            nick_name = info_form.cleaned_data['nick_name']
            gender = info_form.cleaned_data['sex']
            email = info_form.cleaned_data['email']
            mobile = info_form.cleaned_data['mobile']
            birthday = info_form.cleaned_data['birthday']
            signature = info_form.cleaned_data['signature']
            user.nick_name = nick_name
            user.gender = gender
            user.email = email
            user.mobile = mobile
            user.birthday = birthday
            user.signature = signature
            user.save()
            print(nick_name,gender,email,mobile,birthday,signature)
            return render(request, 'personal.html', locals())
        except Exception as e:
        	print('Error: '+ str(e))
        return render(request, 'personal.html', locals())
        