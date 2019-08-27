from django import template
from .models import Category,Tag

register = template.Library()

@register.simple_tag
def get_categories():
	return Category.object.all()

@register.simple_tag
def get_tags():
	return Tag.object.all()