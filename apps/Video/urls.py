from django.urls import path, include
from .views import VideoView, VideoListView

app_name = 'Video'

urlpatterns = [
    path(r'video-list/',VideoListView.as_view(),name='video-list'),
    path(r'video/',VideoView.as_view(),name='video'),
]