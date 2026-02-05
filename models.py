import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

class MusicRecommender:
    def __init__(self, model_path='music_model.pkl', dataset_csv='dataset/music.csv'):
        self.model_path = model_path
        self.dataset_csv = dataset_csv
        self.model = None

    def extract_features(self, file_path, duration=5):
        y, sr = librosa.load(file_path, duration=duration)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)

    def train(self):
        # Load dataset CSV
        df = pd.read_csv(self.dataset_csv)  # columns: filepath,label
        X, y = [], []
        for index, row in df.iterrows():
            features = self.extract_features(row['filepath'])
            X.append(features)
            y.append(row['label'])
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=150)
        self.model.fit(X_train, y_train)
        print("Training accuracy:", self.model.score(X_test, y_test))
        joblib.dump(self.model, self.model_path)

    def predict(self, file_path):
        if self.model is None:
            self.model = joblib.load(self.model_path)
        features = self.extract_features(file_path).reshape(1, -1)
        return self.model.predict(features)[0]

    def recommend(self, label, top_n=5):
        """Return top_n songs from dataset with the same label"""
        df = pd.read_csv(self.dataset_csv)
        return df[df['label'] == label].head(top_n).to_dict('records')
