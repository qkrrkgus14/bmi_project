from django.shortcuts import render

# Create your views here.
# CRUD-L
# Create Read Update Delete List
# 함수형 뷰   :문법적 형식이 함수 - 제네릭뷰가 없거나 내가 기능 하나 하ㅏ를 커스터마징 하고 싶을 때
# 클래스형 뷰 : 문법적 클래스 - 웹 서비스 영역에서 빈번하게 만드는 기능은 제네릭 뷰가 이미 있다.


from django.views.generic.list import ListView
# 제네릭 뷰 -> 모델 베이스뷰
from .models import BMI

class BMIList(ListView):
    model = BMI

from django.views.generic.edit import CreateView,UpdateView,DeleteView
from django.views.generic.detail import DetailView

class BMIDetail(DetailView):
    model = BMI
    fields = ["weight","height"]
    success_url = '/'

class BMICreate(CreateView):
    model = BMI
    fields = ["weight","height"]
    success_url = '/'

class BMIUpdate(UpdateView):
    model = BMI
    fields = ["weight","height"]
    success_url = '/'

class BMIDelete(DeleteView):
    model = BMI
    success_url = '/'


