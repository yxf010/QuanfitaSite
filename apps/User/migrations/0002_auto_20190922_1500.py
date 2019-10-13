# Generated by Django 2.2.1 on 2019-09-22 15:00

import User.models
import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('User', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userprofile',
            name='address',
        ),
        migrations.RemoveField(
            model_name='userprofile',
            name='image',
        ),
        migrations.AddField(
            model_name='userprofile',
            name='avatar',
            field=models.ImageField(default='media/user/default.svg', max_length=500, upload_to=User.models.upload_to, verbose_name='头像'),
        ),
        migrations.AddField(
            model_name='userprofile',
            name='created_time',
            field=models.DateTimeField(auto_now_add=True, default=datetime.datetime(2019, 9, 22, 15, 0, 55, 664976), verbose_name='注册时间'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='userprofile',
            name='email',
            field=models.CharField(default='', max_length=100, verbose_name='邮箱'),
        ),
        migrations.AlterField(
            model_name='userprofile',
            name='password',
            field=models.CharField(max_length=20),
        ),
    ]