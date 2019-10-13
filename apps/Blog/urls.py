from .views import BlogView, BlogListView
from django.conf.urls import url
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = 'Blog'

urlpatterns = [
    path(r'blog-list/', BlogListView.as_view(), name="blog-list"),
    path(r'blog/', BlogView.as_view(), name="blog"),
]