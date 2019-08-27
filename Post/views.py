from django.shortcuts import render
from django.template.loader import get_template
from django.http import HttpResponse
from django.core.paginator import Paginator
from .models import Post,Category,Tag
from django.views import View
import markdown
import os

# Create your views here.
class IndexView(View):
    def get(self,request):
        """
        主页
        :param request:
        :return:
        """
        post = Post.objects.all()

        limit = 3
        paginator = Paginator(post, limit)
        page = request.GET.get('page', 1)

        result = paginator.page(page)
        print(result)
        context = {
            'post_list':result,
            'cate':Category.objects.all(),
            'tag':Tag.objects.all()
        }
        template = get_template('index.html')
        html = template.render(context)
        return HttpResponse(html)

class BlogView(View):
    def get(self,request):
        """
        博客页面
        :param request
        :return:
        """
        blog = request.GET.get('blog')
        post = Post.objects.get(id=blog)
        if post is None:
            return
        name = 'Windows10+Anaconda+TensorFlow(CPU & GPU)环境快速搭建'#post.title
        template = get_template('post.html')
        print(os.path.dirname(__file__))
        docfile = get_template("blogs/{}.md".format(name))
        content = docfile.render()
        '''
        import re
        from urllib.parse import urlencode
        pattern = re.compile(r'(\$\$.*?\$\$)', re.S)
        latex1 = re.sub(pattern, lambda m: '<div align=center><img src="http://latex.codecogs.com/gif.latex?' + urlencode({'':m.group(0).replace('$$','').replace(r'\n','')})[1:]+'"></div>', content, 0)
        pattern2 = re.compile(r'(\$.*?\$)', re.S)
        content = re.sub(pattern2, lambda m: '<img src="'+ 'http://latex.codecogs.com/gif.latex?' +urlencode({'':m.group(0).replace('$','').replace(r'\n','')})[1:]+'">', content, 0)
        content = content.replace('+','')
        '''
        html = template.render({
            'docname': name.replace('_',' '),
            'content':
                markdown.markdown(content,
                                  extensions=[
                                      'markdown.extensions.extra',
                                      'markdown.extensions.codehilite',
                                      'markdown.extensions.toc',
                                      'markdown.extensions.tables',
                                  ])
        })
        return HttpResponse(html)#render(request,template_name='post.html',context=html)
