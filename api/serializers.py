from rest_framework import serializers
from Blog.models import Blog
 
class BlogSerializers(serializers.ModelSerializer):
    class Meta:
        model = Blog     #指定的模型类
        fields = ('id', 'title', 'body', 'created_time', 'excerpt', 'category', 'tags', 'views')   #需要序列化的属性