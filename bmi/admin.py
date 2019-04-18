from django.contrib import admin

# Register your models here.
#관리할 모델을 등록
#관리할 페이지에 대한 커스터마이징 옵셔

from .models import BMI

admin.site.register(BMI)