from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns=[

    path("",views.index,name='index'),
    path("about",views.fabout,name='about'),
    path("userhome",views.userhome,name="userhome"),
    path("login",views.login,name="login"),
    path("Registration",views.Registration,name="Registration"),
    path("upload",views.upload,name='upload'),
    path("splitdata",views.splitdata,name='splitdata'),
    path("modeltrain",views.modeltrain,name='modeltrain'),
    path("prediction",views.prediction,name='prediction')
   
]