from __future__ import absolute_import
from celery import shared_task
from django.apps import apps
import os
import logging
from pathlib import Path
import shutil
import time

logger = logging.getLogger(__name__)

@shared_task(
    bind=True,
    name='process_video_task',
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=3,
    acks_late=True  # Важно для корректной обработки очереди
)
def process_video_task(self, task_id):
    VideoTask = apps.get_model('motion_detector_app', 'VideoTask')
    task = VideoTask.objects.get(id=task_id)
    
    try:
        logger.info(f"Начата обработка задачи {task_id}")
        task.status = 'processing'
        task.save()

        from motion_detector.main import process_video
        
        output_path = Path('media') / 'processed' / f'processed_{Path(task.original_file.name).name}'
        
        # Упрощенный вызов без callback прогресса
        process_video(
            input_path=task.original_file.path,
            output_video_path=str(output_path)
        )
        
        task.processed_file.name = str(output_path.relative_to('media'))
        task.status = 'completed'
        task.save()
        
        logger.info(f"Задача {task_id} успешно завершена")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка обработки задачи {task_id}: {str(e)}")
        task.status = 'failed'
        task.save()
        raise self.retry(exc=e)

def cleanup_files(video_path=None, json_path=None, frames_dir=None):
    """
    Удаляет временные файлы после обработки
    """
    try:
        if video_path and Path(video_path).exists():
            Path(video_path).unlink(missing_ok=True)
        if json_path and Path(json_path).exists():
            Path(json_path).unlink(missing_ok=True)
        if frames_dir and Path(frames_dir).exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")