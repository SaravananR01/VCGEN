from django.db import models

class Teacher(models.Model):
    teacher_id=models.CharField(max_length=8,primary_key=True)
    name=models.CharField(max_length=20)
    email=models.EmailField(null=False)
    password=models.EmailField(null=False)

class Course(models.Model):
    course_id=models.CharField(max_length=8,primary_key=True)
    teacher=models.ForeignKey(Teacher,on_delete=models.CASCADE)
    name=models.CharField(max_length=20)
    hours=models.IntegerField()
    split=models.DecimalField(decimal_places=3,max_digits=10)

class Module(models.Model):
    model_id=models.CharField(max_length=8,primary_key=True)
    course=models.ForeignKey(Course,on_delete=models.CASCADE)
    hours=models.IntegerField()
    name=models.CharField(max_length=100)

class Topics(models.Model):
    topic_id=models.CharField(max_length=8,primary_key=True)
    module=models.ForeignKey(Module,on_delete=models.CASCADE)
    content=models.CharField(max_length=100)
    mapped_skill=models.CharField(max_length=100)
    teacherweight=models.DecimalField(decimal_places=3,max_digits=10)
    studentweight=models.DecimalField(decimal_places=3,max_digits=10)

class Student(models.Model):
    student_id=models.CharField(max_length=8,primary_key=True)
    course=models.ForeignKey(Course,on_delete=models.CASCADE)
    skillsreq=models.CharField(max_length=1000)
