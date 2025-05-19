from django.contrib import admin
from .models import VideoTask

@admin.register(VideoTask)
class VideoTaskAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_file', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('original_file',)
    readonly_fields = ('created_at', 'updated_at')