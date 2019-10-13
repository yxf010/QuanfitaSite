from django.shortcuts import render, redirect
from django.template.loader import get_template
from django.http import HttpResponse
from django.core.paginator import Paginator
from .models import Blog,Category,Tag
from User.models import UserProfile as User
from django.views import View
import markdown
import os

# Create your views here.

class BlogView(View):
    def get(self,request):
        """
        博客页面
        :param request
        :return:
        """
        session = request.session
        try:
            user = User.objects.get(id=session['user_id'])
        except:
            print('User is not login!')
        blog = request.GET.get('id')
        blog = Blog.objects.get(id=blog)
        if blog is None:
            return
        name = blog.title #''Windows10+Anaconda+TensorFlow(CPU & GPU)环境快速搭建''#Blog.title
        template = get_template('blog-single.html')
        print(os.path.dirname(__file__))
        html = template.render({
            'session': request.session,
            'user': user,
            'tag':'Blog',
            'blog_category': blog.category,
            'category': Category.objects.all(),
            'title': name.replace('_',' '),
            'author': blog.author,
            'date': blog.created_time,
            'content': blog.body
        })
        return HttpResponse(html)#render(request,template_name='Blog.html',context=html)

class BlogListView(View):
    """docstring for BlogListView"""
    def get(self, request):
        session = request.session
        try:
            user = User.objects.get(id=session['user_id'])
        except:
            print('User is not login!')
        blogs = Blog.objects.all()

        limit = 5
        paginator = Paginator(blogs, limit)

        page = request.GET.get('page', 1)
        bloglist = paginator.page(page)

        template = get_template('blogs.html')
        html = template.render({
            'session': request.session,
            'user': user,
            'tag':'Blog',
            'category': Category.objects.all(),
            'content': bloglist
        })
        return HttpResponse(html)