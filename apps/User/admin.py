from django.contrib import admin
from .models import UserProfile, EmailVerifyRecord

# Register your models here.
admin.site.register(UserProfile)
admin.site.register(EmailVerifyRecord)

class UserAdmin(admin.ModelAdmin):
	list_display = ('nick_name', 'birthday', 'gender', 'address', 'mobile')
	
		