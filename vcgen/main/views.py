from django.shortcuts import render,redirect

from .models import Teacher,Course,Module,Topics,Student
import random,re,os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer,util
from transformers import pipeline
from django.http import HttpResponse
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO
from .pdf_parser import parse_syllabus_pdf

from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class ScoreDict:
    scores: Dict[str, float]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _as_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str) and x.strip():
        try:
            v = json.loads(x)
            if isinstance(v, list): return v
        except Exception:
            pass
        return [t.strip() for t in re.split(r"[|,]", x) if t.strip()]
    return []

def _prefix_regex(items):
    # represent regex cues as 're:<pattern>'
    out = []
    for it in items or []:
        it = (it or "").strip()
        if it:
            out.append(f"re:{it}")
    return out

def load_skill_repo_csv(csv_path: str):
    """
    Normalizes the new cued CSV into:
      skill: str
      pos:   List[str]          (aliases + keywords + legacy positives, normalized)
      proto: List[str]          (symbol literals + 're:<pattern>' for regex_cues)
      neg:   List[str]          (legacy negatives + 're:<pattern>' for negative_cues)
      student_title, student_desc: passthrough if present
    """
    df = pd.read_csv(csv_path)

    repo = []
    for _, r in df.iterrows():
        skill = str(r.get("skill","")).strip()
        if not skill:
            continue

        # positives / aliases / keywords
        legacy_pos = _as_list(r.get("positives",""))
        aliases    = _as_list(r.get("aliases",""))
        keywords   = _as_list(r.get("keywords",""))
        pos = list(dict.fromkeys([_norm(x) for x in (legacy_pos + aliases + keywords) if x]))

        # strong cues: symbols (literal) + regex (prefixed)
        symbols = _as_list(r.get("symbol_cues",""))
        regexes = _as_list(r.get("regex_cues",""))
        proto   = list(dict.fromkeys([*(x for x in symbols if x), *(_prefix_regex(regexes))]))

        # negatives: legacy + negative regex
        legacy_neg = _as_list(r.get("negatives",""))
        neg_regex  = _as_list(r.get("negative_cues",""))
        neg        = list(dict.fromkeys([_norm(x) for x in legacy_neg if x] + _prefix_regex(neg_regex)))

        repo.append({
            "skill": skill,
            "pos": pos,
            "proto": proto,
            "neg": neg,
            "student_title": r.get("student_title","") or "",
            "student_desc":  r.get("student_desc","") or "",
        })
    return repo

def _has_kw_hit(row: dict, text: str):
    """Return True if any pos/proto hits (supports 're:' patterns and symbol literals)."""
    t_norm = _norm(text)
    t_raw  = text or ""
    # pos (literal on normalized)
    for p in row.get("pos", []):
        if _norm(p) in t_norm:
            return True
    # proto (regex or literal symbol on raw)
    for p in row.get("proto", []):
        if isinstance(p, str) and p.startswith("re:"):
            pat = p[3:].strip()
            try:
                if re.search(pat, t_raw, flags=re.I):
                    return True
            except re.error:
                if _norm(pat) in t_norm:
                    return True
        else:
            if str(p) in t_raw:
                return True
    return False
# ---------- end helpers ----------

# 1) keyword scorer (reuse your helpers)
def keyword_score(text: str, row: Dict, *, cap:int=8, proto_boost:float=1.5) -> float:
    """
    Supports:
      - row['pos'] as literal (normalized) matches
      - row['proto'] elements either 're:<pattern>' (regex on raw text) or literal symbols
      - row['neg'] same as above; any hit → 0.0
    """
    t_norm = _norm(text)
    t_raw  = text or ""
    if not t_norm:
        return 0.0

    # negatives block
    for n in row.get("neg", []):
        if isinstance(n, str) and n.startswith("re:"):
            pat = n[3:].strip()
            try:
                if re.search(pat, t_raw, flags=re.I):
                    return 0.0
            except re.error:
                if _norm(pat) in t_norm:
                    return 0.0
        else:
            if _norm(n) in t_norm:
                return 0.0

    # positives (literal contains on normalized)
    pos = [p for p in row.get("pos", []) if p]
    hits_pos = sum(1 for p in pos if _norm(p) in t_norm)

    # prototypes (regex or literal on raw text)
    proto = [p for p in row.get("proto", []) if p]
    hits_pro = 0
    for p in proto:
        if isinstance(p, str) and p.startswith("re:"):
            pat = p[3:].strip()
            try:
                if re.search(pat, t_raw, flags=re.I):
                    hits_pro += 1
            except re.error:
                if _norm(pat) in t_norm:
                    hits_pro += 1
        else:
            if str(p) in t_raw:
                hits_pro += 1

    hits = hits_pos + proto_boost * hits_pro
    total = min(cap, len(pos) + len(proto)) or 1
    return float(hits) / float(total)

# 2) Hugging Face Zero-shot with BART MNLI
_zs_pipe = None
def zero_shot_scores(text: str, labels: List[str]) -> ScoreDict:
    global _zs_pipe
    try:
        if _zs_pipe is None:
            _zs_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device_map="auto")
        res = _zs_pipe(text, candidate_labels=labels, multi_label=True)
        out = {lab: 0.0 for lab in labels}
        for lab, sc in zip(res["labels"], res["scores"]):
            if lab in out:
                out[lab] = float(sc)
        return ScoreDict(out)
    except Exception:
        # no internet/GPU/etc → uniform
        u = 1.0 / max(1, len(labels))
        return ScoreDict({lab: u for lab in labels})

# 3) Sentence embeddings (cosine sim)
_embed_model = None
def embed_scores(text: str, labels: List[str]) -> ScoreDict:
    global _embed_model
    try:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer, util
            _embed_model = (SentenceTransformer("all-MiniLM-L6-v2"), None)  # cache util in closure below
            _embed_model[0].max_seq_length = 256
        from sentence_transformers import util
        model = _embed_model[0]
        a = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        b = model.encode(labels, normalize_embeddings=True, convert_to_numpy=True)
        sims = (a @ b.T).ravel()  # cosine because normalized
        # map from [-1,1] (theoretically) to [0,1] safeguard
        sims = (sims + 1.0) / 2.0
        return ScoreDict({lab: float(sc) for lab, sc in zip(labels, sims.tolist())})
    except Exception:
        u = 1.0 / max(1, len(labels))
        return ScoreDict({lab: u for lab in labels})

# 4) Blend scores
def blend_scores(zs: ScoreDict, em: ScoreDict, kw: Dict[str, float],
                 labels: List[str], w_zs=0.40, w_emb=0.35, w_kw=0.25) -> Dict[str, float]:
    blended = {}
    for lab in labels:
        z = zs.scores.get(lab, 0.0)
        e = em.scores.get(lab, 0.0)
        k = kw.get(lab, 0.0)
        blended[lab] = w_zs*z + w_emb*e + w_kw*k
    return blended

# 5) Repo-only fallback (no hardcoded skill names) for 0-hit cases
def _tokenize_blob(*parts: str):
    blob = _norm(" ".join(parts))
    return [t for t in re.findall(r"[a-z0-9]+", blob) if len(t) >= 3]

def _fallback_by_repo_only(topic: str, repo_rows: List[Dict]) -> Dict:
    t_tokens = _tokenize_blob(topic)
    best_row, best_score = None, -1e9
    for r in repo_rows:
        meta_tokens = _tokenize_blob(r["skill"], r["student_title"], r["student_desc"])
        score = len(set(t_tokens) & set(meta_tokens)) + 0.1*(len(r["pos"]) + len(r["proto"]))
        if score > best_score:
            best_row, best_score = r, score
    return best_row or repo_rows[0]

# 6) Main HF-blended mapper (uses your CSV loader)
def map_topics_HF_blend(topics: List[str],
                        csv_path: str,
                        min_confidence: float = 0.55,
                        require_kw_hit: bool = False,
                        w_zs=0.40, w_emb=0.35, w_kw=0.25) -> pd.DataFrame:
    """
    - Blends HF zero-shot + embeddings + your keyword score.
    - If `require_kw_hit=True`, only accept mappings that matched at least one repo cue;
      otherwise we allow HF-only mappings too.
    - Uses a repo-only fallback when absolutely nothing matches (so nothing is 'Unmapped').
    """
    repo = load_skill_repo_csv(csv_path)
    labels = [r["skill"] for r in repo]
    out_rows = []

    for t in topics:
        # compute per-label scores
        kw_scores = {r["skill"]: keyword_score(t, r) for r in repo}
        zs = zero_shot_scores(t, labels)
        em = embed_scores(t, labels)
        blended = blend_scores(zs, em, kw_scores, labels, w_zs=w_zs, w_emb=w_emb, w_kw=w_kw)

        # pick best label
        pred = max(blended, key=blended.get)
        conf = float(blended[pred])

        # optionally require at least one keyword/prototype hit
        row_pred = next(r for r in repo if r["skill"] == pred)
        has_kw = _has_kw_hit(row_pred, t)
        kw_hits = []  # optional: collect for logging if you want (costs extra regex scans)


        # If we insist on keyword support but we don't have it, try the next best with a hit
        if require_kw_hit and not has_kw:
            # sort by blended score, pick first with a kw hit
            for alt in sorted(labels, key=lambda L: blended[L], reverse=True):
              row_alt = next(r for r in repo if r["skill"] == alt)
              if _has_kw_hit(row_alt, t):
                pred = alt
                conf = float(blended[pred])
                kw_hits = []   # optional
                has_kw = True
                break


        # if nothing has a hit and confidence is still tiny, use repo-only fallback
        used_fallback = False
        if not has_kw and conf < min_confidence:
            fb = _fallback_by_repo_only(t, repo)
            pred = fb["skill"]
            conf = float(blended.get(pred, 0.0))
            kw_hits = []  # be explicit: this came via fallback
            used_fallback = True

        # gather student fields
        rmap = {r["skill"]: r for r in repo}
        student_title = rmap[pred]["student_title"]
        why = rmap[pred]["student_desc"]

        out_rows.append({
            "topic": t,
            "skill": pred,
            "skill_student": student_title,
            "confidence": round(conf, 4),
            "kw_hits": "; ".join(kw_hits),
            "needs_review": used_fallback or (conf < min_confidence and not has_kw)
        })

    return pd.DataFrame(out_rows)

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
    
    if request.method=='POST':
        faculty=Teacher.objects.filter(teacher_id=request.session['user'])[0]
        newcourse=Course.objects.create(
            course_id=gen_course_id(),
            teacher=faculty,
            name=request.POST['cname'],
            hours=0,
            split=0,
        )
        hours=0
        for i in range(1,8):
            modname=request.POST[f'module_name{i}']
            modhours=request.POST[f'module_hrs{i}']
            modtopics=request.POST[f'topic{i}']
            hours+=int(modhours)
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
            text=text.replace("- ","$$$")
            text=text.replace(" -","$$$")
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
            op=[]
            for topic in topicslist:
                topic=topic.strip(" ")
                if topic[0].isupper()  or not op:
                    op.append(topic)
                else:
                    temp=op.pop()
                    op.append(temp+", "+topic)
            topicslist=op
            '''
            text=modtopics.replace("\n"," ")
            text=text.replace(", and",", ")
            pattern=r' - | – |,'
            topicslist =re.split(pattern, text)'''
            fnresult=map_topics_HF_blend(topicslist,os.path.join(os.path.dirname(__file__), 'skill_repo.csv'))['skill'].to_list()
            for i,topic in enumerate(topicslist):
                #mappedval=ensemble_label(topic)['pred']
                newtopic=Topics.objects.create(
                    topic_id=gen_topic_id(),
                    module=newmod,
                    content=topic,
                    mapped_skill=fnresult[i],
                    teacherweight=0,
                    studentweight=0,
                )
        newcourse.hours=hours
        newcourse.save()
        
        return redirect(f'/settings/{newcourse.course_id}')
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