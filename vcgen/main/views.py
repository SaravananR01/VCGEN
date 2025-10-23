from django.shortcuts import render,redirect

from .models import Teacher,Course,Module,Topics,Student
import random
import re

def gen_teacher_id():
    code="T"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Teacher.objects.filter(teacher_id=code))>0:
        code="T"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code

def gen_course_id():
    code="C"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Teacher.objects.filter(teacher_id=code))>0:
        code="T"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code

def gen_module_id():
    code="M"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Teacher.objects.filter(teacher_id=code))>0:
        code="T"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code

def gen_topic_id():
    code="O"+"".join([str(random.randint(1,9)) for x in range(7)])
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
                    teacher_id=gen_teacher_id(),
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

def results(request,class_id):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    ourclass=Course.objects.filter(course_id=class_id)
    if len(ourclass)<1:
        return redirect('classes')
    else:
        ourclass=ourclass[0]
        data={}
        modules=Module.objects.filter(course=ourclass)
        for mod in modules:
            data[mod.module_id]=[f"{x.content} - {x.teacherweight*mod.hours} + {x.studentweight*mod.hours} = {x.studentweight+x.teacherweight*mod.hours} hours" for x in Topics.objects.filter(module=mod)]
        context['cname']=ourclass.name
        context['modules']= modules
        context['topics']=data
    return render(request,"main/results.html",context=context)

def classes(request):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    context['classes']=Course.objects.filter(teacher=Teacher.objects.filter(teacher_id=request.session['user'])[0])
    return render(request,"main/classes.html",context=context)

def newclass(request):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    if request.method=='POST':
        faculty=Teacher.objects.filter(teacher_id=request.session['user'])[0]
        newcourse=Course.objects.create(
            course_id=gen_course_id(),
            teacher=faculty,
            name=request.POST['cname'],
            hours=request.POST['chours'],
            split=0,
        )
        for i in range(1,8):
            modname=request.POST[f'module_name{i}']
            modhours=request.POST[f'module_hrs{i}']
            modtopics=request.POST[f'topic{i}']

            newmod=Module.objects.create(
                module_id=gen_module_id(),
                course=newcourse,
                hours=modhours,
                name=modname
            )


            text=modtopics.replace("\n"," ")
            text=text.replace(", and",", ")
            pattern=r' - | – |,'
            topicslist =re.split(pattern, text)
            for topic in topicslist:
                newtopic=Topics.objects.create(
                    topic_id=gen_topic_id(),
                    module=newmod,
                    content=topic,
                    mapped_skill="",
                    teacherweight=0,
                    studentweight=0,
                )
        return redirect('classes')
    return render(request,"main/newclass.html")

def settings(request):
    return render(request,"main/settings.html")

def responses(request):
    return render(request,"main/responses.html")

def delete_response(request):
    #add functionality - delete response
    print("response deleted")