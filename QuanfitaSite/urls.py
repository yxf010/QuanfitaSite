"""QuanfitaSite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static
from .views import IndexView, page_not_found, live2d
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^$', IndexView.as_view(), name="index"),
    path(r'', include('Blog.urls',namespace='Blog')),
    path(r'', include('api.urls',namespace='Api')),
    path(r'', include('Video.urls',namespace='Video')),
    path(r'', include('User.urls',namespace='User')),
    url(r'^captcha', include('captcha.urls')),
    url(r'^comments/', include('django_comments.urls')),
    url(r'live2d/',live2d),
    #path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
hander404 = page_not_found