from . import views
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

app_name = 'Post'

urlpatterns = [
    url(r'^$', views.index, name="index"),
    url('blog/', views.post, name="blog"),
]