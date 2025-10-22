from django.contrib import admin

from .models import Teacher,Course,Module,Topics,Student

admin.site.register(Teacher)
admin.site.register(Course)
admin.site.register(Module)
admin.site.register(Topics)
admin.site.register(Student)