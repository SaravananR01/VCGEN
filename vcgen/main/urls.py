from django.urls import path
from . import views

urlpatterns=[
    path("",views.landing,name="landing"),
    path("login/",views.login,name="login"),
    path("signup/",views.signup,name="signup"),
    path("joinclass/",views.joinclass,name="joinclass"),
    path("thankyou/",views.thankyou,name="thankyou"),
    path("survey/",views.survey,name="survey"),
    path("results/",views.results,name="results"),
    path("classes/",views.classes,name="classes"),
    path("newclass/",views.newclass,name="newclass"),
    path("settings/",views.settings,name="settings"),
]