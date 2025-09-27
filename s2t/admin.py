from django.contrib import admin

# Register your models here.
from .models import UserInfo, OpenAITxt, Checklist, M_UserInfo

admin.site.register(UserInfo)
admin.site.register(OpenAITxt)
admin.site.register(Checklist)
admin.site.register(M_UserInfo)