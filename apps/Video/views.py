from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.paginator import Paginator
from django.template.loader import get_template
from .models import Video
from User.models import UserProfile as User
from django.views import View

# Create your views here.
class VideoView(View):
    """docstring for VideoView"""
    def get(self,request):
        session = request.session
        try:
            user = User.objects.get(id=session['user_id'])
        except:
            print('User is not login!')
        video = request.GET.get('id')
        video = Video.objects.get(id=video)

        content = {
                'session': session,
                'user': user,
                'tag': 'Video',
                'category': 'Video',
                'title': video.name,
                'author': 'Quanfita',
                'video_url': video.url,
                'description': video.desc,
                'date': video.add_time
        }

        temp = get_template('video-single.html')
        html = temp.render(content)
        return HttpResponse(html)

class VideoListView(View):
    """docstring for VideoListView"""
    def get(self, request):
        session = request.session
        try:
            user = User.objects.get(id=session['user_id'])
        except:
            print('User is not login!')
        videos = Video.objects.all()

        limit = 5
        paginator = Paginator(videos, limit)

        page = request.GET.get('page', 1)
        videolist = paginator.page(page)

        template = get_template('videos.html')
        html = template.render({
            'session': session,
            'user': user,
            'tag':'Video',
            'content': videolist
        })
        return HttpResponse(html)
        