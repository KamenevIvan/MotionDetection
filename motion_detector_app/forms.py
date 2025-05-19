from django import forms
from .models import VideoTask
from django.core.validators import FileExtensionValidator

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoTask
        fields = ['original_file']
        widgets = {
            'original_file': forms.FileInput(attrs={
                'accept': 'video/*',
                'class': 'form-control'
            })
        }
    
    def clean_original_file(self):
        file = self.cleaned_data.get('original_file')
        if file:
            if file.size > 500 * 1024 * 1024:  # 500MB
                raise forms.ValidationError("File too large. Max size is 500MB")
            ext = file.name.split('.')[-1].lower()
            if ext not in ['mp4', 'avi', 'mov', 'mkv']:
                raise forms.ValidationError("Unsupported file format")
        return file