from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import VideoTask
import os

class VideoTaskTests(TestCase):
    def setUp(self):
        self.test_file = SimpleUploadedFile(
            "test.mp4",
            b"file_content",
            content_type="video/mp4"
        )

    def test_video_task_creation(self):
        task = VideoTask.objects.create(original_file=self.test_file)
        self.assertEqual(task.status, 'queued')
        self.assertTrue(os.path.exists(task.original_file.path))
        
    def tearDown(self):
        for task in VideoTask.objects.all():
            if os.path.exists(task.original_file.path):
                os.remove(task.original_file.path)