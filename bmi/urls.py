from django.urls import path
from . import views



urlpatterns = [
    path("",views.BMIList.as_view(),name="index"),
    path("detail/<int:pk>/",views.BMIDetail.as_view(),name="detail"),
    path("update/<int:pk>/",views.BMIUpdate.as_view(),name="update"),
    path("delete/<int:pk>/",views.BMIDelete.as_view(),name="delete"),
    path("create/",views.BMICreate.as_view(),name="create"),
]
