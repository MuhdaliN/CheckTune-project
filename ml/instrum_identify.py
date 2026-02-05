# ==========================
# Step 1: Import Libraries
# ==========================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Step 2: Create Music Dataset (100 songs)
# ==========================
music_data = pd.DataFrame({
    'title': [f"Song {i}" for i in range(1, 101)],
    'genre': np.random.choice(['Pop', 'Rock', 'Jazz', 'Hip-Hop', 'Classical'], 100),
    'artist': np.random.choice(['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E'], 100),
    'description': np.random.choice([
        'Energetic and upbeat',
        'Relaxing and calm',
        'Melancholic tune',
        'Party vibe',
        'Romantic song',
        'Motivational beat',
        'Soft and soothing',
        'Rocking anthem'
    ], 100)
})

# ==========================
# Step 3: Recommendation Function
# ==========================
def recommend_top_5(selected_title, data):
    # Vectorize the 'description' column
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data['description'])
    
    # Find the index of the selected song
    idx = data[data['title'] == selected_title].index[0]
    
    # Compute cosine similarity with all other songs
    similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get top 5 similar songs (exclude the selected song itself)
    similar_indices = similarity_scores.argsort()[-6:-1][::-1]
    
    # Return the top 5 recommended songs with details
    return data.iloc[similar_indices][['title', 'genre', 'artist']]

# ==========================
# Step 4: Select a song and get recommendations
# ==========================
selected_song = 'Song 10'  # You can change this to any song from the list
top_5_recommendations = recommend_top_5(selected_song, music_data)

# ==========================
# Step 5: Display results
# ==========================
print(f"Selected Song: {selected_song}")
print("\nTop 5 Recommended Songs:")
print(top_5_recommendations)
