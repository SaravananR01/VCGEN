from django.shortcuts import render,redirect
from django.conf import settings
from .models import Teacher,Course,Module,Topics,Student
import random,re,os
import pandas as pd
import numpy as np
from django.http import HttpResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO
from .pdf_parser import parse_syllabus_pdf
from scipy.optimize import linprog
from groq import Groq
from dotenv import load_dotenv
import json, os, re
import time

from dataclasses import dataclass
from typing import List, Dict, Any
import json

load_dotenv()

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client

def build_skill_block_for_prompt(repo: list) -> str:
    lines = []
    for i, r in enumerate(repo, 1):
        skill = r["skill"]
        desc  = r.get("description", "") or r.get("student_desc", "") or skill
        first_sentence = desc.split(".")[0].strip() + "."
        lines.append(f"{i}. {skill}: {first_sentence}")
    return "\n".join(lines)

def load_skill_repo_csv(csv_path):
    df = pd.read_csv(csv_path,encoding="cp1252")
    repo = []
    for _, r in df.iterrows():
        repo.append({
            "skill":             r["skill"],
            "description": r.get("description", "") or r.get("student_desc", ""),
            "survey_tier":       r.get("survey_tier", ""),        
            "department_tags":   r.get("department_tags", ""),    
        })
    return repo

def map_all_topics_with_groq(
    modules_data: list[dict],
    subject_context: str,
    csv_path: str,
    batch_size: int = 3,        
) -> dict[str, dict]:

    repo         = load_skill_repo_csv(csv_path)
    client       = get_groq_client()
    valid_skills = [r["skill"] for r in repo]

    skill_block = build_skill_block_for_prompt(repo)

    all_topics_flat = []
    for mod in modules_data:
        for topic in mod["topics"]:
            all_topics_flat.append((mod["module_name"], topic))

    batches = [
        all_topics_flat[i:i + batch_size]
        for i in range(0, len(all_topics_flat), batch_size)
    ]

    result = {}

    for batch_num, batch in enumerate(batches):
        syllabus_block = "\n".join(
            f"  - {topic}" for topic in batch
        )

        prompt = f"""Map each course topic to 1-3 academic skills from the list below.

COURSE: {subject_context}

SKILLS (use exact names only):
{skill_block}

TOPICS ([Module] Topic):
{syllabus_block}

Return JSON only:
{{"mappings": [{{"topic": "<exact topic name>", "skills": ["Skill 1"], "confidence": {{"Skill 1": 0.9}}}}]}}

Rules:
- Use only skill names from the list above, copied exactly.
- 1 to 3 skills per topic.
- Every topic must appear once."""

        raw = ""
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    #model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a curriculum analyst. Return valid JSON only. No markdown, no explanation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )

                finish_reason = response.choices[0].finish_reason
                raw           = response.choices[0].message.content.strip()

                print(f"[Groq] batch={batch_num+1}/{len(batches)} "
                      f"attempt={attempt+1} "
                      f"finish_reason={finish_reason} "
                      f"response_len={len(raw)}")

                if finish_reason == "length":
                    print(f"[Groq] WARNING: batch {batch_num+1} truncated.")

                break 

            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    wait = 60 if attempt == 0 else 120
                    print(f"[Groq] Rate limit hit. Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    print(f"[Groq ERROR] batch={batch_num+1} attempt={attempt+1}: {e}")
                    break

        try:
            parsed   = json.loads(raw)
            mappings = parsed.get("mappings", [])

            for entry in mappings:
                topic  = entry.get("topic", "")
                skills = [s for s in entry.get("skills", []) if s in valid_skills]
                conf   = {k: v for k, v in entry.get("confidence", {}).items()
                          if k in valid_skills}

                if not skills:
                    print(f"[MAPPING FAIL] '{topic}' returned unknown skills: {entry.get('skills')}. Flagged for review.")
                    result[topic] = {"skill": ["UNMAPPED"], "confidence": {}}
                    continue

                result[topic] = {"skill": skills, "confidence": conf}

        except (json.JSONDecodeError, Exception) as e:
            print(f"[Groq JSON ERROR] batch={batch_num+1}: {e}")
            print(f"[Groq] raw was: {raw[:300]}")
            for _, topic in batch:
                kw_scores = {r["skill"]: keyword_score(topic, r) for r in repo}
                best      = max(kw_scores, key=kw_scores.get)
                result[topic] = {"skill": [best], "confidence": {best: 0.5}}

        if batch_num < len(batches) - 1:
            time.sleep(30)

    return result


def compute_lp_student_weights(all_topics, modules_map, skill_demand, course_hours, split):
    student_pool = float(course_hours) * (1.0 - float(split))
    n = len(all_topics)

    if n == 0 or not skill_demand or student_pool <= 0:
        return {t.topic_id: 0.0 for t in all_topics}

    topic_demands = []
    for topic in all_topics:
        mapped = topic.mapped_skill
        if isinstance(mapped, str):
            skills = [mapped]
        elif isinstance(mapped, list):
            skills = mapped
        else:
            skills = []

        try:
            confidences = topic.skill_confidence if topic.skill_confidence else {}
        except AttributeError:
            confidences = {}

        weighted_sum = 0.0
        weight_total = 0.0
        for skill in skills:
            conf = float(confidences.get(skill, 1.0))
            demand = float(skill_demand.get(skill, 0.0))
            weighted_sum += demand * conf
            weight_total += conf

        topic_demand = weighted_sum / weight_total if weight_total > 0 else 0.0
        topic_demands.append(topic_demand)

    topic_demands = np.array(topic_demands, dtype=float)

    if topic_demands.sum() == 0:
        uniform_w = 1.0 / n
        return {t.topic_id: uniform_w for t in all_topics}

    floors = []
    for topic in all_topics:
        mod = modules_map[topic.module_id]
        teacher_w = float(topic.teacherweight)
        proportional_share = teacher_w * student_pool
        floor = proportional_share * 0.30 if teacher_w > 0 else 0.0
        floors.append(floor)

    floors = np.array(floors, dtype=float)

    floor_sum = floors.sum()
    if floor_sum > student_pool:
        floors = floors * (student_pool / floor_sum) * 0.95

    c = -topic_demands

    A_eq = np.ones((1, n))
    b_eq = np.array([student_pool])

    module_ids = list({topic.module_id for topic in all_topics})
    A_ub_rows = []
    b_ub_rows = []

    for mid in module_ids:
        mod = modules_map[mid]
        mod_topic_indices = [i for i, t in enumerate(all_topics) if t.module_id == mid]
        if not mod_topic_indices:
            continue
        proportional_mod_share = (float(mod.hours) / max(1, sum(
            modules_map[m].hours for m in module_ids
        ))) * student_pool
        cap = proportional_mod_share * 1.5

        row = np.zeros(n)
        for idx in mod_topic_indices:
            row[idx] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(cap)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    bounds = [(floors[i], student_pool) for i in range(n)]

    try:
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )

        if result.success:
            allocated_hours = result.x
        else:
            print(f"[LP WARNING] {result.message} — falling back to proportional.")
            total_demand = topic_demands.sum() or 1.0
            allocated_hours = (topic_demands / total_demand) * student_pool
            for i in range(n):
                if allocated_hours[i] < floors[i]:
                    allocated_hours[i] = floors[i]

    except Exception as e:
        print(f"[LP ERROR] {e} — falling back to proportional.")
        total_demand = topic_demands.sum() or 1.0
        allocated_hours = (topic_demands / total_demand) * student_pool

    weights = {}
    for i, topic in enumerate(all_topics):
        weights[topic.topic_id] = float(allocated_hours[i]) / student_pool

    return weights

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
        elif not reqclass[0].accepting_response:
            context['error']="Responses Closed"
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
        csv_path = os.path.join(os.path.dirname(__file__), 'skill_repo.csv')
        repo = load_skill_repo_csv(csv_path)

        skill_to_tier = {r["skill"]: r.get("survey_tier", r["skill"]) for r in repo}

        survey_tiers = set()
        for mod in mods:
            for topic in Topics.objects.filter(module=mod):
                mapped = topic.mapped_skill
                if not mapped:
                    continue
                if isinstance(mapped, str):
                    mapped = [mapped]
                for s in mapped:
                    tier = skill_to_tier.get(s, s)
                    survey_tiers.add(tier)

        skills = sorted(survey_tiers) 
        print(skills)
        context['skills'] = skills
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

            
            students = Student.objects.filter(course=ourclass)
            skillvotes = {}
            total_students = 0
            for s in students:
                prefs = [p.strip() for p in s.skillsreq.split(",") if p.strip()]
                if not prefs:
                    continue
                total_students += 1
                for p in prefs:
                    skillvotes[p] = skillvotes.get(p, 0) + 1
            
            tier_map = {}
            for r in repo:
                tier = r.get("survey_tier", r["skill"])
                tier_map.setdefault(tier, []).append(r["skill"])

            skill_demand = {}
            for survey_label, vote_count in skillvotes.items():
                for tier2_skill in tier_map.get(survey_label, [survey_label]):
                    skill_demand[tier2_skill] = skill_demand.get(tier2_skill, 0) + (vote_count / max(1, total_students))
            mods = Module.objects.filter(course=ourclass)
            all_topics = []
            modules_map = {}
            for m in mods:
                modules_map[m.module_id] = m
                for topic in Topics.objects.filter(module=m):
                    all_topics.append(topic)

            lp_weights = compute_lp_student_weights(
                all_topics=all_topics,
                modules_map=modules_map,
                skill_demand=skill_demand,
                course_hours=ourclass.hours,
                split=float(ourclass.split),
            )

            for topic in all_topics:
                topic.studentweight = lp_weights.get(topic.topic_id, 0.0)
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

def download_dayplan_pdf(request, class_id):
    if 'user' not in request.session:
        return redirect('login')

    ourclass=Course.objects.filter(course_id=class_id).first()
    if not ourclass:
        return redirect('classes')

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
            self.time = teachertime + studenttime

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
    tempcontent,tempmod="",""
    lastcontent,lastmod="",""
    for mod in ourmods:
        for topic in mod.topicslist:
            tempcontent+=topic.topic.content+", "
            lastcontent=topic.topic.content
            if mod.mod.name not in tempmod:
                tempmod+=mod.mod.name+", "
                lastmod=mod.mod.name
            cur += topic.time
            while cur > 1:
                tempcontent=tempcontent.rstrip(", ")
                tempmod=tempmod.rstrip(", ")
                slots.append(slot(tempcontent, tempmod))
                if cur==1:
                    cur=0
                    tempcontent, tempmod = "", ""
                elif cur>1:
                    cur-=1
                    tempcontent=lastcontent + ", "
                    tempmod=lastmod + ", "
    tempcontent=tempcontent.rstrip(", ")
    tempmod=tempmod.rstrip(", ")
    slots.append(slot(tempcontent, tempmod))

    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer, pagesize=A4)
    styles=getSampleStyleSheet()
    elements=[]

    elements.append(Paragraph(f"<b>{ourclass.name} - {class_id}</b>", styles['Title']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Day-wise Plan", styles['Heading2']))
    elements.append(Spacer(1, 10))

    for i, s in enumerate(slots, 1):
        elements.append(Paragraph(f"<b>Hour {i}:</b> {s.mod}", styles['Heading3']))
        elements.append(Paragraph(s.content, styles['Normal']))
        elements.append(Spacer(1, 8))

    doc.build(elements)

    pdf=buffer.getvalue()
    buffer.close()

    response=HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="dayplan_{class_id}.pdf"'
    return response

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
    
    if request.method=='POST' and 'syllabus_pdf' in request.FILES:
        parsed_modules=parse_syllabus_pdf(request.FILES['syllabus_pdf'])
        context['parsed_modules']=parsed_modules
        for i in parsed_modules:
            print(i)
        print("Length of PDF: ",len(parsed_modules))
        return render(request,"main/newclass.html",context)
    # if request.method=='POST':
    #     faculty=Teacher.objects.filter(teacher_id=request.session['user'])[0]
    #     newcourse=Course.objects.create(
    #         course_id=gen_course_id(),
    #         teacher=faculty,
    #         name=request.POST['cname'],
    #         hours=0,
    #         split=0,
    #     )
    #     hours=0
    #     for i in range(1,8):
    #         modname=request.POST[f'module_name{i}']
    #         modhours=request.POST[f'module_hrs{i}']
    #         modtopics=request.POST[f'topic{i}']
    #         hours+=int(modhours)
    #         newmod=Module.objects.create(
    #             module_id=gen_module_id(),
    #             course=newcourse,
    #             hours=modhours,
    #             name=modname
    #         )
    #         text=modtopics.replace("\n"," ")
    #         text=text.replace(", and",", ")
    #         text=text.replace(" - ","$$$")
    #         text=text.replace(" – ","$$$")
    #         text=text.replace("- ","$$$")
    #         text=text.replace(" -","$$$")
    #         brackettrue=False
    #         op=""
    #         for x in text:
    #             if x=="," and not brackettrue:
    #                 op+="$$$"
    #             else:
    #                 op+=x
                
    #             if x=="(":
    #                 brackettrue=True
    #             elif x==")":
    #                 brackettrue=False
    #         text=op
    #         topicslist=text.split("$$$")
    #         op=[]
    #         for topic in topicslist:
    #             topic=topic.strip(" ")
    #             if topic[0].isupper()  or not op:
    #                 op.append(topic)
    #             else:
    #                 temp=op.pop()
    #                 op.append(temp+", "+topic)
    #         topicslist=op
    #         '''
    #         text=modtopics.replace("\n"," ")
    #         text=text.replace(", and",", ")
    #         pattern=r' - | – |,'
    #         topicslist =re.split(pattern, text)'''
    #         subject=request.POST['cname']
    #         print(subject)
    if request.method == 'POST':
        faculty   = Teacher.objects.filter(teacher_id=request.session['user'])[0]
        newcourse = Course.objects.create(
            course_id=gen_course_id(),
            teacher=faculty,
            name=request.POST['cname'],
            hours=0,
            split=0,
        )

        hours       = 0
        modules_data = []  

        for i in range(1, 8):
            modname   = request.POST[f'module_name{i}']
            modhours  = request.POST[f'module_hrs{i}']
            modtopics = request.POST[f'topic{i}']
            hours    += int(modhours)

            newmod = Module.objects.create(
                module_id=gen_module_id(),
                course=newcourse,
                hours=modhours,
                name=modname
            )

            text = modtopics.replace("\n", " ")
            text = text.replace(", and", ", ")
            text = text.replace(" - ", "$$$").replace(" – ", "$$$")
            text = text.replace("- ", "$$$").replace(" -", "$$$")
            brackettrue = False
            op = ""
            for x in text:
                if x == "," and not brackettrue:
                    op += "$$$"
                else:
                    op += x
                if x == "(":
                    brackettrue = True
                elif x == ")":
                    brackettrue = False
            topicslist = [t.strip() for t in op.split("$$$") if t.strip()]
            final_topics = []
            for topic in topicslist:
                topic = topic.strip()
                if not topic:
                    continue
                if topic[0].isupper() or not final_topics:
                    final_topics.append(topic)
                else:
                    final_topics[-1] += ", " + topic

            modules_data.append({
                "module_name": modname,
                "topics":      final_topics,
                "mod_obj":     newmod,      
            })

        subject    = request.POST['cname']
        csv_path   = os.path.join(os.path.dirname(__file__), 'skill_repo.csv')
        topic_map = map_all_topics_with_groq(modules_data, subject, csv_path)
        print(topic_map)
        for mod_data in modules_data:
            for topic in mod_data["topics"]:
                mapping    = topic_map.get(topic, {"skill": ["Foundations & Theory"], "confidence": {}})
                skills     = mapping["skill"]
                skill_conf = mapping["confidence"]

                newtopic = Topics.objects.create(
                    topic_id=gen_topic_id(),
                    module=mod_data["mod_obj"],
                    content=topic,
                    mapped_skill=skills,
                    teacherweight=0,
                    studentweight=0,
                )
                newtopic.skill_confidence = skill_conf
                newtopic.save()

        newcourse.hours = hours
        newcourse.save()
        return redirect(f'/settings/{newcourse.course_id}')
        #     fnresult=map_topics_HF_blend(topicslist,subject,os.path.join(os.path.dirname(__file__), 'skill_repo.csv'))['skill'].to_list()
        #     fnresult=map_topics_via_bloom(topicslist,subject,os.path.join(os.path.dirname(__file__), 'skill_repo.csv'))['skill'].to_list()
        #     for i,topic in enumerate(topicslist):
        #         #mappedval=ensemble_label(topic)['pred']
        #         newtopic=Topics.objects.create(
        #             topic_id=gen_topic_id(),
        #             module=newmod,
        #             content=topic,
        #             mapped_skill=fnresult[i],
        #             teacherweight=0,
        #             studentweight=0,
        #         )
            
        #     ALt
        #     bloom_df = map_topics_via_bloom(
        #         topicslist, subject,
        #         os.path.join(os.path.dirname(__file__), 'skill_repo.csv')
        #     )
        #     for i, topic in enumerate(topicslist):
        #         row = bloom_df.iloc[i]
        #         skills = row['skill']         
        #         confidences = row['confidence']

        #         skill_conf = {}
        #         bloom_to_skill_map = BLOOM_TO_SKILL
        #         for bloom_label, score in confidences.items():
        #             for skill in bloom_to_skill_map.get(bloom_label, []):
        #                 if skill in skills:
        #                     skill_conf[skill] = max(skill_conf.get(skill, 0.0), score)

        #         newtopic = Topics.objects.create(
        #             topic_id=gen_topic_id(),
        #             module=newmod,
        #             content=topic,
        #             mapped_skill=skills,       
        #             teacherweight=0,
        #             studentweight=0,
        #         )
        #         try:
        #             newtopic.skill_confidence = skill_conf
        #             newtopic.save()
        #         except Exception:
        #             pass 

        #     #primary
        #     hf_df = map_topics_HF_blend_2(
        #         topicslist, subject,
        #         os.path.join(os.path.dirname(__file__), 'skill_repo.csv')
        #     )
        #     for i, topic in enumerate(topicslist):
        #         row = hf_df.iloc[i]

        #         skills = row['skill']
        #         if isinstance(skills, str):
        #             skills = [skills]
        #         elif not isinstance(skills, list):
        #             skills = list(skills)

        #         confidences = row['confidence'] 
        #         skill_conf = {skill: float(confidences.get(skill, 0.0)) for skill in skills}

        #         newtopic = Topics.objects.create(
        #             topic_id=gen_topic_id(),
        #             module=newmod,
        #             content=topic,
        #             mapped_skill=skills,
        #             teacherweight=0,
        #             studentweight=0,
        #         )
        #         try:
        #             newtopic.skill_confidence = skill_conf
        #             newtopic.save()
        #         except Exception:
        #             pass
        # newcourse.hours=hours
        # newcourse.save()
        
        # return redirect(f'/settings/{newcourse.course_id}')
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

def closeresponses(request,class_id):
    context={}
    if 'user' not in request.session:
        return redirect('login')
    else:
        toclose=Course.objects.filter(course_id=class_id)
        toclose=toclose[0]
        if toclose.accepting_response:
            toclose.accepting_response=False
            toclose.save()
    return redirect('/classes')

def update_progress(request, class_id):
    context = {}
    if 'user' not in request.session:
        return redirect('login')
    else:
        context['name'] = Teacher.objects.filter(teacher_id=request.session['user'])[0].name

    ourclass = Course.objects.filter(course_id=class_id)
    if len(ourclass) < 1:
        return redirect('classes')

    ourclass = ourclass[0]
    modules = Module.objects.filter(course=ourclass)

    class cusModule:
        def __init__(self, mod):
            self.name = mod.name
            self.hours = mod.hours
            self.topics = []

        def addtopic(self, topic):
            self.topics.append(topic)

    datamodules = []
    for mod in modules:
        curmod = cusModule(mod)
        for topic in Topics.objects.filter(module=mod):
            curmod.addtopic(topic)
        datamodules.append(curmod)

    context['cid'] = class_id
    context['cname'] = ourclass.name
    context['modules'] = datamodules

    if request.method == "POST":
        total_hours = int(request.POST.get('total_hours', ourclass.hours))
        completed_hours = int(request.POST.get('completed_hours', 0))
        completed_topic_ids = set(request.POST.getlist('completed_topics'))

        remaining_hours = max(0, total_hours - completed_hours)

        all_topic_objs = []
        modules_map = {}
        for mod in modules:
            modules_map[mod.module_id] = mod
            for topic in Topics.objects.filter(module=mod):
                all_topic_objs.append(topic)

        for topic in all_topic_objs:
            if topic.topic_id in completed_topic_ids:
                topic.teacherweight = 0
                topic.studentweight = 0
                topic.save()

        pending_topics = [t for t in all_topic_objs if t.topic_id not in completed_topic_ids]

        students = Student.objects.filter(course=ourclass)
        skillvotes = {}
        totalvotes = 0
        for s in students:
            for p in s.skillsreq.split(","):
                p = p.strip()
                if p:
                    skillvotes[p] = skillvotes.get(p, 0) + 1
                    totalvotes += 1
        skill_demand = {a: b / totalvotes for a, b in skillvotes.items()} if totalvotes else {}

        if pending_topics and skill_demand: 
            lp_weights = compute_lp_student_weights(
                all_topics=pending_topics,
                modules_map=modules_map,
                skill_demand=skill_demand,
                course_hours=remaining_hours,
                split=float(ourclass.split),
            )
            for topic in pending_topics:
                topic.studentweight = lp_weights.get(topic.topic_id, 0.0)
                topic.save()
        elif pending_topics:
            for topic in pending_topics:
                topic.studentweight = 0.0
                topic.save()

        ourclass.hours = total_hours
        ourclass.save()

        return redirect(f'/results/{class_id}')

    return render(request, "main/dynamic.html", context=context)

def analytics(request, class_id):
    context = {}
    if 'user' not in request.session:
        return redirect('login')
    else:
        context['name'] = Teacher.objects.filter(teacher_id=request.session['user'])[0].name

    ourclass = Course.objects.filter(course_id=class_id).first()
    if not ourclass:
        return redirect('classes')

    modules = Module.objects.filter(course=ourclass)
    all_topics = []
    for mod in modules:
        for topic in Topics.objects.filter(module=mod):
            all_topics.append((mod, topic))

    students = Student.objects.filter(course=ourclass)
    context['cid']           = class_id
    context['cname']         = ourclass.name
    context['total_students']= students.count()
    context['total_topics']  = len(all_topics)
    context['total_modules'] = modules.count()
    context['course_hours']  = ourclass.hours

    skill_votes = {}
    total_votes = 0
    for s in students:
        for p in s.skillsreq.split(","):
            p = p.strip()
            if p:
                skill_votes[p] = skill_votes.get(p, 0) + 1
                total_votes += 1

    sorted_skills = sorted(skill_votes.items(), key=lambda x: x[1], reverse=True)
    skill_labels  = [s[0] for s in sorted_skills]
    skill_counts  = [s[1] for s in sorted_skills]

    context['skill_labels'] = skill_labels
    context['skill_votes']  = skill_counts
    context['top_skill']    = skill_labels[0] if skill_labels else "—"

    all_mapped_skills = set()
    for _, topic in all_topics:
        mapped = topic.mapped_skill or []
        if isinstance(mapped, str):
            mapped = [mapped]
        all_mapped_skills.update(mapped)

    hm_skill_order = skill_labels + [s for s in sorted(all_mapped_skills) if s not in skill_labels]
    context['skill_labels'] = hm_skill_order  

    def _conf_color(val):
        """Return hex bg and text color for a 0-1 confidence value."""
        if val <= 0:
            return "#f0f8f6", "#c0d4d0"         
        r = int(0xda + (0x3d - 0xda) * val)
        g = int(0xf0 + (0x9e - 0xf0) * val)
        b = int(0xee + (0x85 - 0xee) * val)
        bg = f"#{r:02x}{g:02x}{b:02x}"
        text = "#fff" if val > 0.55 else "#2e3a35"
        return bg, text

    heatmap_rows = []
    for mod, topic in all_topics:
        mapped    = topic.mapped_skill or []
        if isinstance(mapped, str):
            mapped = [mapped]
        conf_dict = topic.skill_confidence or {}

        cells = []
        for skill in hm_skill_order:
            if skill in mapped:
                raw = conf_dict.get(skill, 0.5)
                val = round(float(raw), 2)
            else:
                val = 0.0
            bg, fg = _conf_color(val)
            cells.append({
                "skill":      skill,
                "val":        val,
                "has_val":    val > 0,   
                "color":      bg,
                "text_color": fg,
            })

        heatmap_rows.append({
            "topic": topic.content,
            "cells": cells,
        })

    context['heatmap_rows'] = heatmap_rows

    max_mod_hrs = 1
    module_alloc = []
    for mod in modules:
        t_hrs, s_hrs = 0.0, 0.0
        for topic in Topics.objects.filter(module=mod):
            t_hrs += round(float(topic.teacherweight) * float(mod.hours) * float(ourclass.split), 2)
            s_hrs += round(float(topic.studentweight) * float(ourclass.hours) * (1 - float(ourclass.split)), 2)
        total = t_hrs + s_hrs
        if total > max_mod_hrs:
            max_mod_hrs = total
        module_alloc.append({
            "name":        mod.name,
            "teacher_hrs": round(t_hrs, 1),
            "student_hrs": round(s_hrs, 1),
            "total":       round(total, 1),
        })

    for m in module_alloc:
        m['teacher_pct'] = round(m['teacher_hrs'] / max_mod_hrs * 100, 1)
        m['student_pct'] = round(m['student_hrs'] / max_mod_hrs * 100, 1)

    context['module_alloc'] = module_alloc

    topic_hours = []
    for mod, topic in all_topics:
        t_hrs = round(float(topic.teacherweight) * float(mod.hours) * float(ourclass.split), 2)
        s_hrs = round(float(topic.studentweight) * float(ourclass.hours) * (1 - float(ourclass.split)), 2)
        topic_hours.append({"name": topic.content, "hours": round(t_hrs + s_hrs, 2)})

    topic_hours.sort(key=lambda x: x['hours'], reverse=True)
    top_n = topic_hours[:12]
    max_th = max((t['hours'] for t in top_n), default=1)
    for t in top_n:
        t['pct'] = round(t['hours'] / max_th * 100, 1)

    context['top_topics'] = top_n

    return render(request, "main/analytics.html", context=context)