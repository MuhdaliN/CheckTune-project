from django.db import models
from django.contrib.auth.models import User

class User_db(models.Model):
    name_db = models.CharField(max_length=100, blank=True, null=True)
    email_db = models.EmailField(unique=True, blank=True, null=True)
    password_db = models.CharField(max_length=255)

    class Meta:
        verbose_name_plural = "Users"

    def __str__(self):
        return self.name_db or "Unknown"



class Album(models.Model):
    title = models.CharField(max_length=200)
    artist = models.CharField(max_length=200)
    genre = models.CharField(max_length=100) 

    def __str__(self):
        return self.title


class AudioUpload(models.Model):
    user = models.ForeignKey(User_db,on_delete=models.CASCADE,null=True,blank=True)
    audio = models.FileField(upload_to='audios/')
    predicted_label = models.CharField(max_length=100, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    all_scores = models.JSONField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.audio.name


