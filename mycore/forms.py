from django import forms
from .models import AudioUpload
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class AudioUploadForm(forms.ModelForm):
    class Meta:
        model = AudioUpload
        fields = ['audio']
        
class UploadForm(forms.Form):
    file = forms.FileField(label="Choose WAV file")

class SearchForm(forms.Form):
    query = forms.CharField(label="Search label", max_length=100)



