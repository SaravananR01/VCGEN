from django.shortcuts import render,redirect

from .models import Teacher,Course,Module,Topics,Student
import random
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer,util
from transformers import pipeline
#import torch

SKILLS = {
    "Problem Solving": "finding problems, thinking of solutions, and building programs that work step by step",
    "Algorithmic Thinking": "breaking tasks into clear steps, planning methods, and comparing ways to solve problems efficiently",
    "Analytical Thinking": "examining how programs and algorithms behave, spotting issues, and improving them",
    "Efficiency Awareness": "understanding how fast and how much memory a program uses and improving its performance",
    "Critical Thinking": "questioning results, comparing different approaches, and deciding what works best",
    "Logical Reasoning": "using clear rules and reasoning to prove how programs or systems behave",
    "Programming Basics": "writing simple programs, using loops, functions, and variables in common languages",
    "Data Handling": "organizing and using information with lists, trees, or tables for better performance",
    "Modular Design": "dividing programs into smaller reusable parts and keeping code organized",
    "Object-Oriented Thinking": "creating programs using real-world models with objects, classes, and inheritance",
    "Software Design": "planning and organizing how programs are structured and how parts work together",
    "Formal Language Understanding": "knowing how computers read, understand, and translate code through grammars or compilers",
    "Code Improvement": "cleaning, simplifying, and optimizing code to make it faster and easier to read",
    "Computer System Understanding": "learning how computers process data, store information, and run instructions",
    "Database Design": "structuring information, linking related data, and keeping it well organized for storage",
    "Data Retrieval": "getting information quickly using indexes, search methods, and sorting techniques",
    "Smart Querying": "writing database questions that run faster and give the right results",
    "System Management": "understanding how systems use memory, processing, and files to work efficiently",
    "Task Coordination": "managing multiple operations or programs running at the same time safely and smoothly",
    "Reliability Awareness": "ensuring systems recover from errors and keep working correctly",
    "Networking Basics": "understanding how computers share data safely and efficiently over networks",
    "Information Security": "keeping systems and information safe from unauthorized access or attacks",
    "Creative Problem Solving": "finding unique or alternate ways to reach a goal or improve performance",
    "Hardware Interaction": "understanding how computers connect to sensors and other devices to perform tasks",
    "Mathematical Thinking": "using logic, patterns, and numbers to design and analyze algorithms",
    "Project Planning": "organizing work, setting goals, and completing programming or research projects",
    "Requirements Understanding": "knowing what a system needs to do and how to translate those needs into design plans",
    "Process Understanding": "knowing how software or systems evolve through different stages of development",
    "Testing and Validation": "checking that programs work correctly and improving them through feedback",
    "Quality Awareness": "making sure work follows good practices and produces reliable results"
}


LABELS = list(SKILLS.keys())
LABEL_TEXTS = [f"{k}: {v}" for k, v in SKILLS.items()]

#device = 0 if torch.cuda.is_available() else -1

zshot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")#, device=device)
encodings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#encodings = encodings.to(device)
skills_embs = encodings.encode(LABELS,normalize_embeddings=True)

def zero_shot_label(text,labels=LABELS):
  out = zshot(text,labels,multi_label=False)
  return dict(pred=out["labels"][0],conf=float(out["scores"][0]),scores=dict(zip(out["labels"],out["scores"])))

def embed_label(text):
  v = encodings.encode([text],normalize_embeddings=True)[0]
  cs = util.cos_sim(v,skills_embs).flatten().tolist() # Corrected variable name to skills_embs
  idx = int(np.argmax(cs))
  return dict(pred=LABELS[idx],conf=float(cs[idx]),scores=dict(zip(LABELS,cs)))

def ensemble_label(text, w_zs=0.5, w_emb=0.5, tau=0.55):
    zs = zero_shot_label(text)
    em = embed_label(text)
    # Normalize cosine to 0..1 for blending
    em01 = {k: (v+1)/2 for k,v in em["scores"].items()}
    labels = LABELS
    blended = {k: w_zs*zs["scores"][k] + w_emb*em01[k] for k in labels}
    pred = max(blended, key=blended.get)
    conf = blended[pred]
    return dict(pred=pred, conf=conf, review=conf<tau, zs=zs, em=em, blended=blended)

def gen_teacher_id():
    code="T"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Teacher.objects.filter(teacher_id=code))>0:
        code="T"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code

def gen_course_id():
    code="C"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Course.objects.filter(course_id=code))>0:
        code="C"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code

def gen_module_id():
    code="M"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Module.objects.filter(module_id=code))>0:
        code="M"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code

def gen_topic_id():
    code="O"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Topics.objects.filter(topic_id=code))>0:
        code="O"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code

def gen_student_id():
    code="S"+"".join([str(random.randint(1,9)) for x in range(7)])
    while len(Student.objects.filter(student_id=code))>0:
        code="S"+"".join([str(random.randint(1,9)) for x in range(7)])
    return code


def landing(request):
    request.session.clear()
    request.session.flush()
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
                request.session.set_expiry(6000)
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
    context={}
    if request.method=="POST":
        code=request.POST['code']
        name=request.POST['name']
        regno=request.POST['regno']
        reqclass=Course.objects.filter(course_id=code)
        if len(reqclass)<1:
            context['error']="Invalid code!"
        elif re.match(r'[0-9]{2}[a-zA-Z]{3}[0-9]{4,}',regno)==None:
            context['error']="Invalid Registration Number"
        else:
            student=Student.objects.create(
                student_id=gen_student_id(),
                name=name,
                regno=regno,
                course=reqclass[0],
                skillsreq=""
            )
            request.session['student']=student.student_id
            request.session.modified=True
            request.session.set_expiry(6000)
            return redirect(f'/survey/{code}')
    return render(request,"main/joinclass.html",context=context)

def thankyou(request):
    request.session.clear()
    request.session.flush()
    return render(request,"main/thankyou.html")

def survey(request,class_id):
    context={}
    ourclass=Course.objects.filter(course_id=class_id)
    if 'student' not in request.session:
        return redirect('login')
    else:
        student=Student.objects.filter(student_id=request.session['student'])[0]
        context['name']=student.name
    if len(ourclass)<1:
        return redirect("/")
    else:
        ourclass=ourclass[0]
        context['cid']=class_id
        context['cname']=ourclass.name
        mods=Module.objects.filter(course=ourclass)
        skills=[]
        for mod in mods:
            skills.extend([x.mapped_skill for x in Topics.objects.filter(module=mod) if x.mapped_skill ])
        skills=list(set(skills))
        skills.sort()
        context['skills']=skills
        if request.method=="POST":
            preferredskills=""
            print(request.POST)
            for skill in skills:
                query=f'value_{skill}'
                if query not in request.POST:
                    continue
                else:
                    preferredskills+=skill+","
            
            preferredskills=preferredskills.rstrip(",")
            student.skillsreq=preferredskills
            student.save()

            
            students=Student.objects.filter(course=ourclass)
            skillvotes={}
            totalvotes=0
            for s in students:
                pref=s.skillsreq.split(",")
                for p in pref:
                    if p in skillvotes:
                        skillvotes[p]+=1
                    else:
                        skillvotes[p]=1
                    totalvotes+=1

            skillweights={a:b/totalvotes for a,b in skillvotes.items()}
            topicstoskills={}
            mods=Module.objects.filter(course=ourclass)
            for m in mods:
                for topic in Topics.objects.filter(module=m):
                    if topic.mapped_skill not in topicstoskills:
                        topicstoskills[topic.mapped_skill]={'count':1,'topics':[topic.content]}
                    else:
                        topicstoskills[topic.mapped_skill]['count']+=1
                        topicstoskills[topic.mapped_skill]['topics'].append(topic.content)
            print(skillweights,topicstoskills)
            for m in mods:
                for topic in Topics.objects.filter(module=m):
                    mappedskill=topic.mapped_skill
                    if mappedskill not in skillweights:
                        topic.studentweight=0
                    else:
                        topic.studentweight=skillweights[mappedskill]/topicstoskills[mappedskill]['count']
                    topic.save()

            return redirect('/thankyou')
    return render(request,"main/survey.html",context=context)

def results(request,class_id):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        context['name']=Teacher.objects.filter(teacher_id=request.session['user'])[0].name
    ourclass=Course.objects.filter(course_id=class_id)
    if len(ourclass)<1:
        return redirect('classes')
    else:
        ourclass=ourclass[0]
        data={}
        modules=Module.objects.filter(course=ourclass)

        class cusModule:
            def __init__(self,mod):
                self.mod=mod
                self.topicslist=[]
                self.modtime=0

            def addtopic(self,topic):
                self.topicslist.append(topic)
                self.modtime+=topic.time

        class cusTopic:

            def __init__(self,topic,teachertime,studenttime):
                self.topic=topic
                self.teachertime=teachertime
                self.studenttime=studenttime
                self.time=teachertime+studenttime
                #self.cuslabel=ensemble_label(self.topic.content)['pred']



        ourmods=[]
        for mod in modules:
            newmod=cusModule(mod)

            #data[mod.module_id]=[]
            for topic in Topics.objects.filter(module=mod):
                teachertime=round(topic.teacherweight*mod.hours*ourclass.split,2)
                studenttime=round(topic.studentweight*ourclass.hours*(100-(ourclass.split*100))/100,2)
                newtopic=cusTopic(topic,teachertime,studenttime)
                newmod.addtopic(newtopic)

                
                #data[mod.module_id].append(f"{topic.content} - {teachertime} + {studenttime} = {teachertime+studenttime} hours")
            ourmods.append(newmod)


        context['cid']=class_id
        context['cname']=ourclass.name
        context['modules']= ourmods#modules
        #context['topics']=data
    return render(request,"main/results.html",context=context)

def dayplan(request,class_id):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        context['name']=Teacher.objects.filter(teacher_id=request.session['user'])[0].name
    ourclass=Course.objects.filter(course_id=class_id)
    if len(ourclass)<1:
        return redirect('classes')
    else:
        ourclass=ourclass[0]
        data={}
        modules=Module.objects.filter(course=ourclass)

        class cusModule:
            def __init__(self,mod):
                self.mod=mod
                self.topicslist=[]
                self.modtime=0

            def addtopic(self,topic):
                self.topicslist.append(topic)
                self.modtime+=topic.time

        class cusTopic:

            def __init__(self,topic,teachertime,studenttime):
                self.topic=topic
                self.teachertime=teachertime
                self.studenttime=studenttime
                self.time=teachertime+studenttime
                #self.cuslabel=ensemble_label(self.topic.content)['pred']

        ourmods=[]
        for mod in modules:
            newmod=cusModule(mod)
            for topic in Topics.objects.filter(module=mod):
                teachertime=round(topic.teacherweight*mod.hours*ourclass.split,2)
                studenttime=round(topic.studentweight*ourclass.hours*(100-(ourclass.split*100))/100,2)
                newtopic=cusTopic(topic,teachertime,studenttime)
                newmod.addtopic(newtopic)
            ourmods.append(newmod)

        class slot:

            def __init__(self,content,mod):
                self.content=content
                self.mod=mod
        slots=[]
        cur=0
        tempcontent=""
        tempmod=""
        lastcontent=""
        lastmod=""
        for mod in ourmods:
            for topic in mod.topicslist:
                tempcontent+=topic.topic.content+", "
                lastcontent=topic.topic.content
                if mod.mod.name not in tempmod:
                    tempmod+=mod.mod.name+", "
                    lastmod=mod.mod.name
                cur+=topic.time
                #print(tempcontent,tempmod)
                while cur>1:
                    tempcontent=tempcontent.rstrip(", ")
                    tempmod=tempmod.rstrip(", ")
                    newslot=slot(tempcontent,tempmod)
                    #print(tempcontent,tempmod)
                    slots.append(newslot)
                    if cur==1:
                        cur=0
                        tempcontent=""
                        tempmod=""
                    elif cur>1:
                        cur-=1
                        tempcontent=lastcontent+", "
                        tempmod=lastmod+", "
        tempcontent=tempcontent.rstrip(", ")
        tempmod=tempmod.rstrip(", ")
        newslot=slot(tempcontent,tempmod)
        #print(tempcontent,tempmod)
        slots.append(newslot)
        context['cid']=class_id
        context['cname']=ourclass.name
        context['slots']=slots
    return render(request,"main/dayplan.html",context=context)

def classes(request):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        context['name']=Teacher.objects.filter(teacher_id=request.session['user'])[0].name
    context['classes']=Course.objects.filter(teacher=Teacher.objects.filter(teacher_id=request.session['user'])[0])
    return render(request,"main/classes.html",context=context)

def newclass(request):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        context['name']=Teacher.objects.filter(teacher_id=request.session['user'])[0].name
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
            text=text.replace(" - ","$$$")
            text=text.replace(" – ","$$$")
            brackettrue=False
            op=""
            for x in text:
                if x=="," and not brackettrue:
                    op+="$$$"
                else:
                    op+=x
                
                if x=="(":
                    brackettrue=True
                elif x==")":
                    brackettrue=False
            text=op
            topicslist=text.split("$$$")
            '''
            text=modtopics.replace("\n"," ")
            text=text.replace(", and",", ")
            pattern=r' - | – |,'
            topicslist =re.split(pattern, text)'''

            for topic in topicslist:
                mappedval=ensemble_label(topic)['pred']
                newtopic=Topics.objects.create(
                    topic_id=gen_topic_id(),
                    module=newmod,
                    content=topic,
                    mapped_skill=mappedval,
                    teacherweight=0,
                    studentweight=0,
                )
        return redirect(f'settings/{newcourse.course_id}')
    return render(request,"main/newclass.html")

def settings(request,class_id):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        context['name']=Teacher.objects.filter(teacher_id=request.session['user'])[0].name
    ourclass=Course.objects.filter(course_id=class_id)
    if len(ourclass)<1:
        return redirect('classes')
    else:
        class cusModule:
            def __init__(self,mod):
                self.mod=mod
                self.topicslist=[]
                
            
            def addtopic(self,topic):
                self.topicslist.append(topic)

        ourclass=ourclass[0]
        datamodules=[]
        modules=Module.objects.filter(course=ourclass)
        for mod in modules:
            curmod=cusModule(mod)
            for topic in Topics.objects.filter(module=mod):
                curmod.addtopic(topic)
            datamodules.append(curmod)

        context['cid']=class_id
        context['cname']=ourclass.name
        context['modules']= datamodules
        if request.method=="POST":
            coursesplit=int(request.POST[f'sv_hoursplit'])
            ourclass.split=coursesplit/100
            ourclass.save()
            for mod in datamodules:
                weightsum=0
                for topic in mod.topicslist:
                    weightsum+=int(request.POST[f'weight_{topic.topic_id}'])
                for topic in mod.topicslist:
                    topic.teacherweight=int(request.POST[f'weight_{topic.topic_id}'])/weightsum
                    topic.save()
            return redirect(f'/results/{class_id}')
            
    return render(request,"main/settings.html",context=context)

def responses(request,class_id):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        context['name']=Teacher.objects.filter(teacher_id=request.session['user'])[0].name
    ourclass=Course.objects.filter(course_id=class_id)
    if len(ourclass)<1:
        return redirect('classes')
    else:
        ourclass=ourclass[0]
        class cusstudent:
            def __init__(self,student):
                self.student=student
                self.skills=student.skillsreq.split(",")
        context['cname']=ourclass.name
        context['cid']=ourclass.course_id
        context['responses']=[cusstudent(student) for student in Student.objects.filter(course=ourclass)]
    return render(request,"main/responses.html",context=context)

def delete_response(request,class_id,student_id):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        todelete=Student.objects.filter(student_id=student_id)
        if todelete:
            todelete=todelete[0]
            todelete.delete()
    return redirect(f'/responses/{class_id}')

def delete_class(request,class_id):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        todelete=Course.objects.filter(course_id=class_id)
        if todelete:
            todelete=todelete[0]
            todelete.delete()
    return redirect('/classes')