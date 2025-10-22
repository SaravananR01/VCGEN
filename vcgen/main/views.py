from django.shortcuts import render

# Create your views here.

def landing(request):
    return render(request,"main/landing.html")

def login(request):
    return render(request,"main/login.html")

def signup(request):
    return render(request,"main/signup.html")

def joinclass(request):
    return render(request,"main/joinclass.html")

def thankyou(request):
    return render(request,"main/thankyou.html")

def survey(request):
    return render(request,"main/survey.html")

def results(request):
    return render(request,"main/results.html")