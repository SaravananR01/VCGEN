from django.shortcuts import render,redirect

from .models import Teacher,Course,Module,Topics,Student
import random

def gen_t_id():
    code="T"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Teacher.objects.filter(teacher_id=code))>0:
        code="T"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code

def landing(request):
    return render(request,"main/landing.html")

def login(request):
    context={}
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user=Teacher.objects.filter(email=email)
        if not user:
            context["error"]="No such user found!"
        else:
            user=user[0]
            if not user.check_password(password):
                context["error"]="Wrong password!"
            else:
                request.session['user']=user.teacher_id
                request.session.modified=True
                request.session.set_expiry(600)
                return redirect('classes')   

    return render(request,'main/login.html',context=context)

def signup(request):
    context={}
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        password=request.POST['password']
        confirmpassword=request.POST['confpass']

        if password!=confirmpassword:
            context['error']="Passwords do not match!"
        else:
            accs=Teacher.objects.filter(email=email)
            if len(accs)>0:
                context['error']="Your account already exists!"
            elif "vit.ac.in" not in email:
                print(email)
                context['error']="You are not eligible for an account."
            else:
                newteacher=Teacher.objects.create(
                    teacher_id=gen_t_id(),
                    name=name,
                    email=email
                )
                newteacher.set_password(password)
                newteacher.save()
                return redirect('login')


    return render(request,"main/signup.html",context=context)

def joinclass(request):
    return render(request,"main/joinclass.html")

def thankyou(request):
    return render(request,"main/thankyou.html")

def survey(request):
    return render(request,"main/survey.html")

def results(request):
    return render(request,"main/results.html")