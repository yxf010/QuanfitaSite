# Generated by Django 2.2.1 on 2019-09-20 03:47

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Category',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Tag',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Blog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100)),
                ('body', models.TextField()),
                ('author', models.CharField(max_length=20)),
                ('created_time', models.DateTimeField(verbose_name=django.utils.timezone.now)),
                ('excerpt', models.CharField(blank=True, max_length=200)),
                ('views', models.PositiveIntegerField(default=0)),
                ('category', models.ForeignKey(on_delete=True, to='Blog.Category')),
                ('tags', models.ManyToManyField(blank=True, to='Blog.Tag')),
            ],
        ),
    ]
