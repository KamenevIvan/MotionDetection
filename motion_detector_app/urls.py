from django.urls import path
from django.urls import re_path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_video, name='upload_video'),
    path('tasks/<int:task_id>/status/', views.task_status, name='task_status'),
    path('tasks/<int:task_id>/delete/', views.delete_task, name='delete_task'),
    re_path(r'^media/(?P<path>.*)$', views.video_view, name='media'),
     path('tasks/recent/', views.recent_tasks, name='recent_tasks'),
]