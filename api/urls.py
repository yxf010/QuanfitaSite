from django.urls import path
from rest_framework import routers
from .views import BlogViewSet, BlogDetail

app_name = 'Api'

urlpatterns = [
    path('api/', BlogViewSet.as_view()),
    path('api/<int:pk>/', BlogDetail.as_view())
]