from .models import Photo
from django.contrib import admin

admin.site.register(Photo)

# Register your models here.
class PhotoAdmin(admin.ModelAdmin):
	list_display = ('name', 'tags', 'desc', 'add_time')