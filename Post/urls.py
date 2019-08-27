from .views import IndexView,BlogView
from django.conf.urls import url
from django.conf import settings
from django.conf.urls.static import static

app_name = 'Post'

urlpatterns = [
    url(r'^$', IndexView.as_view(), name="index"),
    url('blog/', BlogView.as_view(), name="blog"),
]