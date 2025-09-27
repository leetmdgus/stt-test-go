from django.contrib import admin

# Register your models here.
from .models import UserInfo, OpenAITxt, Checklist

admin.site.register(UserInfo)
admin.site.register(OpenAITxt)
admin.site.register(Checklist)