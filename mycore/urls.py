from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('contact/', views.contact, name='contact'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('upload/', views.upload_audio, name='upload'),
    path('result/<int:audio_id>/', views.result_view, name='result'),
    path('login/', views.login_signup, name='login_signup'),
    path('chatbot_api/', views.chatbot_api, name='chatbot_api'),
    path('recommend/', views.music_recommendation, name='music_recommendation'),
    path('logout/', views.log_out, name='logout'),
]
