import os
import csv
import json
import pandas as pd
import numpy as np
from collections import Counter
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from django.urls import reverse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.core.mail import EmailMessage
from .models import User_db, AudioUpload
from .forms import AudioUploadForm
from ml.predict import predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django import forms

music_data = pd.read_csv(r'C:\Users\refre\OneDrive\Desktop\CheckTunes(R)\ml\music_dataset.csv')
if 'description' not in music_data.columns:
    music_data['description'] = music_data['Title'] + ' ' + music_data['Artist'] + ' ' + music_data['Genre']

class MusicSelectForm(forms.Form):
    song = forms.ChoiceField(
        choices=[(row['Title'], f"{row['Title']} - {row['Artist']} ({row['Genre']})") for _, row in music_data.iterrows()],
        label="Select a Song",
        widget=forms.Select(attrs={'class': 'form-control'})
    )

def home(request):
    if not request.session.get('logged_user') and request.COOKIES.get('logged_user'):
        request.session['logged_user'] = request.COOKIES.get('logged_user')
        user = User_db.objects.filter(email_db=request.session['logged_user']).first()
        if user:
            request.session['name'] = user.name_db

    csv_path = os.path.join(settings.BASE_DIR, "mycore", "static", "music_dataset.csv")
    songs = []

    with open(csv_path, newline='', encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            songs.append({
                "Title": row["Title"],
                "Artist": row["Artist"],
                "Genre": row["Genre"]
            })

    genre = request.GET.get("genre", "All")
    recommended = []

    if request.session.get("logged_user"):
        genres = [s["Genre"] for s in songs if s["Genre"]]
        if genres:
            top_genre = Counter(genres).most_common(1)[0][0]
            recommended = [s for s in songs if s["Genre"] == top_genre][:12]

    return render(request, "home.html", {
        "songs": songs,
        "genre": genre,
        "recommended": recommended
    })

def contact(request):
    if request.method == "POST":
        name = request.POST.get('name')
        phone = request.POST.get('phone_no')
        from_email = request.POST.get('email')
        subject = request.POST.get('subject')
        message = request.POST.get('message')
        to_email = "muhammedalinowshad25@gmail.com"
        email_body = f"Name: {name}\nEmail: {from_email}\nPhone: {phone}\n\nMessage:\n{message}\n\n— CheckTune Support"

        try:
            email = EmailMessage(subject, email_body, from_email, [to_email], reply_to=[from_email])
            email.send()
            messages.success(request, "Your message was sent successfully!")
        except Exception as e:
            print("Email sending error:", e)
            messages.error(request, "Failed to send message. Try again later.")

    return render(request, 'contact.html')

def chatbot(request):
    return render(request, 'chatbot.html')

@csrf_exempt
def chatbot_api(request):
    if request.method != "POST":
        return JsonResponse({"reply": "Invalid request"}, status=400)

    data = json.loads(request.body)
    msg = data.get("message", "").lower()
    upload_url = reverse('upload')
    login_url = reverse('login_signup')

    if "upload" in msg:
        reply = f"Upload your audio file: <a class='chat-link' href='{upload_url}'>Upload Audio</a>"
    elif "login" in msg:
        reply = f"Login here: <a class='chat-link' href='{login_url}'>Login</a>"
    elif "register" in msg:
        reply = f"Create an account: <a class='chat-link' href='{login_url}'>Register</a>"
    elif "format" in msg:
        reply = "Supported audio format: WAV (less than 20 seconds)."
    else:
        reply = "I'm here to help! Ask me anything about CheckTune 😊"

    return JsonResponse({"reply": reply})

@csrf_exempt
def login_signup(request):
    if request.method == "POST":
        form_type = request.POST.get("form_type")

        if form_type == "signup":
            name = request.POST.get('name')
            email = request.POST.get('email')
            password = request.POST.get('password')

            if User_db.objects.filter(email_db=email).exists():
                messages.error(request, "Email already registered!")
                return redirect('login_signup')

            User_db.objects.create(name_db=name, email_db=email, password_db=password)
            messages.success(request, "Signup successful! Please log in.")
            return redirect('login_signup')

        elif form_type == "login":
            email = request.POST.get('email')
            password = request.POST.get('password')
            remember = request.POST.get('remember') == 'on'

            try:
                user = User_db.objects.get(email_db=email)
                if password == user.password_db:
                    request.session['logged_user'] = user.email_db
                    request.session['name'] = user.name_db
                    response = redirect('home')
                    if remember:
                        response.set_cookie('logged_user', user.email_db, max_age=86400)
                    return response
                else:
                    messages.error(request, "Incorrect password.")
                    return redirect('login_signup')
            except User_db.DoesNotExist:
                messages.error(request, "Email not found!")
                return redirect('login_signup')

    return render(request, "login&register.html")

def upload_audio(request):
    if not request.session.get('logged_user') and not request.COOKIES.get('logged_user'):
        messages.error(request, "You must be logged in to upload audio.")
        return redirect('login_signup')

    if not request.session.get('logged_user') and request.COOKIES.get('logged_user'):
        request.session['logged_user'] = request.COOKIES.get('logged_user')
        user = User_db.objects.filter(email_db=request.session['logged_user']).first()
        if user:
            request.session['name'] = user.name_db

    form = AudioUploadForm()

    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            user_email = request.session.get('logged_user')
            user = User_db.objects.filter(email_db=user_email).first()
            
            if not user:
                messages.error(request, "Invalid session. Please log in again.")
                return redirect('login_signup')
            
            obj.user = user
            obj.save()

            file_path = obj.audio.path
            model_path = os.path.join(settings.BASE_DIR, 'ml', 'model_checktune.h5')
            labels_map = os.path.join(settings.BASE_DIR, 'ml', 'labels_map.json')

            try:
                label, conf, all_scores = predict(file_path, model_path=model_path, labels_map=labels_map)
                obj.predicted_label = label
                obj.confidence = conf
                obj.all_scores = json.dumps({k: float(v)*100 for k, v in all_scores.items()})
                obj.save()
                return redirect('result', audio_id=obj.id)
            except Exception as e:
                form.add_error(None, f"Prediction error: {e}")

    return render(request, 'upload.html', {'form': form})

def log_out(request):
    request.session.flush()
    response = redirect('home')
    response.delete_cookie('logged_user')
    return response

def result_view(request, audio_id):
    audio = get_object_or_404(AudioUpload, id=audio_id)
    scores = json.loads(audio.all_scores) if audio.all_scores else {}

    if scores:
        top_label = max(scores, key=scores.get)
        top_conf = scores[top_label]
    else:
        top_label = audio.predicted_label
        top_conf = audio.confidence

    context = {
        'audio': audio,
        'scores': scores,
        'top_label': top_label,
        'top_conf': top_conf,
    }
    return render(request, 'result.html', context)

def recommend_top_5(selected_title):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(music_data['description'])
    idx = music_data[music_data['Title'] == selected_title].index[0]
    similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = similarity_scores.argsort()[-6:-1][::-1]
    
    top5_list = []
    for i in similar_indices:
        top5_list.append({
            'title': music_data.iloc[i]['Title'],
            'artist': music_data.iloc[i]['Artist'],
            'genre': music_data.iloc[i]['Genre'],
            'score': round(similarity_scores[i]*100, 1)
        })
    return top5_list

def music_recommendation(request):
    prediction = None
    top5_list = None
    stats = {
        'total_songs': len(music_data),
        'top_match': 0,
        'database_coverage': 0,
        'avg_top_3': 0
    }

    if request.method == "POST":
        form = MusicSelectForm(request.POST)
        if form.is_valid():
            selected_song = form.cleaned_data['song']
            prediction = f"Top 5 recommendations for {selected_song}:"
            top5_list = recommend_top_5(selected_song)
            
            # Calculate statistics
            if top5_list:
                stats['top_match'] = top5_list[0]['score']
                stats['database_coverage'] = (5 / len(music_data)) * 100  # Always 5 recommendations
                if len(top5_list) >= 3:
                    avg = sum(item['score'] for item in top5_list[:3]) / 3
                    stats['avg_top_3'] = round(avg, 1)
                else:
                    stats['avg_top_3'] = sum(item['score'] for item in top5_list) / len(top5_list)
    else:
        form = MusicSelectForm()

    return render(request, 'music_recommendation.html', {
        'form': form,
        'prediction': prediction,
        'top5_list': top5_list,
        'stats': stats
    })