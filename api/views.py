from rest_framework import generics
from Blog.models import Blog
from .serializers import BlogSerializers


class BlogViewSet(generics.ListAPIView):
    # 指定结果集并设置排序
    queryset = Blog.objects.all()
    # 指定序列化的类
    serializer_class = BlogSerializers

class BlogDetail(generics.RetrieveAPIView):
    queryset = Blog.objects.all()
    serializer_class = BlogSerializers