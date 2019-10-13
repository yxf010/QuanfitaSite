from django.shortcuts import render
from django.template.loader import get_template
from django.http import HttpResponse
from django.core.paginator import Paginator
from Blog.models import Blog,Category,Tag
from Video.models import Video
from User.models import UserProfile as User
from django.views import View


class IndexView(View):
    def get(self,request):
        """
        主页
        :param request:
        :return:
        """
        blog = Blog.objects.all().order_by('created_time')
        video = Video.objects.all().order_by('add_date')
        
        page = request.GET.get('page', 1)

        limit = 4
        paginator = Paginator(blog, limit)
        blog_list = paginator.page(page)

        limit = 5
        paginator = Paginator(video, limit)
        video_list = paginator.page(page)

        print('login' if request.user.is_authenticated else 'not login')
        if request.user.is_authenticated and len(request.session.keys()) > 3:
            print(request.session.keys())
            context = {
                'session': request.session,
                'user': User.objects.get(id=request.session['user_id']),
                'tag': 'Home',
                'blog_list': blog_list,
                'video_list': video_list,
                'category':Category.objects.all(),
                'tags':Tag.objects.all()
            }
        else:
            context = {
                #'session': request.session,
                #'user': User.objects.get(id=request.session['user_id']),
                'tag': 'Home',
                'blog_list': blog_list,
                'video_list': video_list,
                'category':Category.objects.all(),
                'tags':Tag.objects.all()
            }
        template = get_template('index.html')
        html = template.render(context)
        return HttpResponse(html)

def page_not_found(request,**kwargs):
    return render_to_response('404.html')

def live2d(request):
    return render(request, 'live2d.html')