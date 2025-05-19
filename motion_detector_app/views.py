from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import VideoTask
from .tasks import process_video_task
import os
from django.conf import settings
from django.views.decorators.http import require_http_methods, require_GET
from django.views.static import serve
from celery.result import AsyncResult
from django.core.files.storage import default_storage
import time
import logging
import time
import datetime
logger = logging.getLogger(__name__)

def index(request):
    tasks = VideoTask.objects.all().order_by('-created_at')
    processing_tasks = tasks.filter(status='processing').count()
    
    context = {
        'tasks': tasks,
        'processing_tasks': processing_tasks
    }
    return render(request, 'motion_detector_app/index.html', context)

def check_queue(request):
    """Проверка статуса очереди"""
    processing = VideoTask.objects.filter(status='processing').count()
    pending = VideoTask.objects.filter(status='pending').count()
    
    return JsonResponse({
        'processing': processing,
        'pending': pending,
        'total': processing + pending
    })

def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        # Проверка расширения файла
        ext = os.path.splitext(video_file.name)[1][1:].lower()
        if ext not in ['mp4', 'avi', 'mov', 'mkv']:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid file type. Allowed: mp4, avi, mov, mkv'
            }, status=400)
        
        # Создаем задачу без привязки к пользователю
        task = VideoTask.objects.create(original_file=video_file)
        process_video_task.delay(task.id)
        
        return JsonResponse({
            'status': 'success',
            'task_id': task.id
        })
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

def task_status(request, task_id):
    try:
        task = VideoTask.objects.get(id=task_id)
        
        if task.status == 'completed' and task.progress != 100:
            task.progress = 100
            task.save()
            
        return JsonResponse({
            'status': task.status,
            'progress': task.progress,
            'processed_url': task.processed_file.url if task.processed_file else None
        })
        
    except VideoTask.DoesNotExist:
        return JsonResponse({'status': 'not_found'}, status=404)
    
def video_view(request, path):
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    return serve(request, os.path.basename(file_path), os.path.dirname(file_path))

def delete_task(request, task_id):
    try:
        task = VideoTask.objects.get(id=task_id)
        
        # Удаляем файлы с повторными попытками
        def safe_delete(path):
            for _ in range(3):  # 3 попытки
                try:
                    if path and default_storage.exists(path):
                        default_storage.delete(path)
                        return True
                except Exception as e:
                    logger.warning(f"Delete attempt failed: {str(e)}")
                    time.sleep(0.5)  # Задержка между попытками
            return False
        
        # Удаляем связанные файлы
        file_paths = [
            task.original_file.path if task.original_file else None,
            task.processed_file.path if task.processed_file else None
        ]
        
        for path in file_paths:
            safe_delete(path)
        
        # Удаляем саму задачу
        task.delete()
        
        return JsonResponse({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {str(e)}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
def recent_tasks(request):
    """Возвращает последние задачи"""
    tasks = VideoTask.objects.order_by('-created_at')
    data = [{
        'id': task.id,
        'status': task.status,
        'status_display': task.get_status_display(),
        'processed_url': task.processed_file.url if task.processed_file else None
    } for task in tasks]
    return JsonResponse(data, safe=False)

@require_GET
def task_updates(request):
    """Long-polling endpoint для получения обновлений задач"""
    last_update = request.GET.get('last_update', 0)
    timeout = 30  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Проверяем изменения после last_update
        updated_tasks = VideoTask.objects.filter(
            updated_at__gt=datetime.fromtimestamp(float(last_update))
        )
            
        if updated_tasks.exists():
            data = {
                'tasks': [{
                    'id': task.id,
                    'status': task.status,
                    'status_display': task.get_status_display(),
                    'processed_url': task.processed_file.url if task.processed_file else None,
                } for task in updated_tasks],
                'deleted_tasks': []  # Можно добавить логику для удаленных задач
            }
            return JsonResponse(data)
            
        time.sleep(1)  # Проверяем каждую секунду
    
    return JsonResponse({'tasks': [], 'deleted_tasks': []})