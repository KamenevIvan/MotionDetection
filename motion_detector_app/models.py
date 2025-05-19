from django.db import models
import os
from django.conf import settings
from pathlib import Path
import shutil

def get_upload_path(instance, filename):
    # Убираем user_id из пути, если он не нужен
    return f'uploads/{filename}'

def get_processed_path(instance, filename):
    # Убираем user_id из пути, если он не нужен
    return f'processed/{filename}'

class VideoTask(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default='pending'
    )

    # Убираем ForeignKey на пользователя, если он не используется
    original_file = models.FileField(upload_to=get_upload_path)
    processed_file = models.FileField(upload_to=get_processed_path, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    progress = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def save(self, *args, **kwargs):
        if hasattr(self, 'progress'):
            self.progress = min(max(self.progress, 0), 100)  # Ограничиваем 0-100%
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{os.path.basename(self.original_file.name)} - {self.status}"

    def get_status_display(self):
        return dict(self.STATUS_CHOICES).get(self.status, self.status)
    
    def delete(self, *args, **kwargs):
        self._delete_related_files()
        super().delete(*args, **kwargs)
    
    def _delete_related_files(self):
        """Удаляет все связанные с задачей файлы"""
        if self.original_file:
            original_path = Path(self.original_file.path)
            if original_path.exists():
                original_path.unlink(missing_ok=True)
        
        if self.processed_file:
            processed_path = Path(self.processed_file.path)
            if processed_path.exists():
                processed_path.unlink(missing_ok=True)
        
        frames_dir = Path(settings.MEDIA_ROOT) / 'temp_frames' / f'task_{self.id}'
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)

class Meta:
    ordering = ['-created_at']  # Сортировка по дате создания