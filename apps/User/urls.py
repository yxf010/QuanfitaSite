from django.urls import path, include
from django.conf.urls import url
from .views import RegisterView, LoginView, logout, PersonSpaceView

app_name = 'User'

urlpatterns = [
    path(r'login/', LoginView.as_view(), name='login'),
    path(r'register/', RegisterView.as_view(), name='register'),
    path(r'logout/', logout, name='logout'),
    path(r'space/', PersonSpaceView.as_view(), name='space')
]