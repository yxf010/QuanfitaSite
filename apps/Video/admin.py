from django.contrib import admin
from .models import Video

# Register your models here.
admin.site.register(Video)

class VideoAdmin(admin.ModelAdmin):
	list_display = ('aid','name', 'tags', 'url', 'cover', 'desc', 'add_time')
	