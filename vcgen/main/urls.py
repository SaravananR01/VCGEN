from django.urls import path
from . import views

urlpatterns=[
    path("",views.landing,name="landing"),
    path("login/",views.login,name="login"),
    path("signup/",views.signup,name="signup"),
    path("joinclass/",views.joinclass,name="joinclass"),
    path("thankyou/",views.thankyou,name="thankyou"),
    path("survey/<str:class_id>",views.survey,name="survey"),
    path("results/<str:class_id>",views.results,name="results"),
    path("dayplan/<str:class_id>",views.dayplan,name="dayplan"),
    path('download_dayplan/<str:class_id>/', views.download_dayplan_pdf, name='download_dayplan'),
    path("classes/",views.classes,name="classes"),
    path("newclass/",views.newclass,name="newclass"),
    path("deleteclass/<str:class_id>",views.delete_class,name="delete_class"),
    path("settings/<str:class_id>",views.settings,name="settings"),
    path("responses/<str:class_id>",views.responses,name="responses"),
    path("deleteresponse/<str:class_id>/<str:student_id>",views.delete_response,name="delete_response"),
    path("closeresponses/<str:class_id>",views.closeresponses,name="closeresponses"),
    path("updateprogress/<str:class_id>",views.update_progress,name="update_progress"),
    path('analytics/<str:class_id>/', views.analytics, name='analytics'),
]