from django.apps import AppConfig

class MotionDetectorAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'motion_detector_app'

    def ready(self):
        pass