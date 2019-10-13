from django.contrib import admin
from .models import Blog, Tag, Category
# Register your models here.
admin.site.register(Blog)
admin.site.register(Tag)
admin.site.register(Category)

class BlogAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'created_time', 'modified_time', 'views')

class TagAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')

class Category(admin.ModelAdmin):
    list_display = ('id', 'name')