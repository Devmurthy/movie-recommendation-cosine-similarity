from flask import Flask, jsonify, render_template, request
import pandas as pd
import requests
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()
OMDB_API_KEY = os.getenv('OMDB_API_KEY')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
POSTER_FALLBACK = '/static/poster-placeholder.svg'

def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'movies.csv'))
    features = ['genres', 'keywords', 'cast', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')
    df['combined_features'] = df.apply(lambda row: ' '.join([str(row[feat]) for feat in features]), axis=1)
    return df

movies = load_data()

def build_vectorizer(df):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(df['combined_features'])
    return cv, count_matrix

vectorizer, count_matrix = build_vectorizer(movies)

def get_movie_index(title):
    result = movies[movies['title'].str.lower() == title.strip().lower()]
    if not result.empty:
        return result.index[0]
    return None

@lru_cache(maxsize=2048)
def fetch_poster_omdb(title, year=None):
    if not OMDB_API_KEY:
        return POSTER_FALLBACK
    params = {
        't': title,
        'apikey': OMDB_API_KEY
    }
    if year:
        params['y'] = str(year)
    try:
        response = requests.get('https://www.omdbapi.com/', params=params, timeout=4)
        response.raise_for_status()
        data = response.json()
        poster_url = data.get('Poster')
        if data.get('Response') == 'True' and poster_url and poster_url != 'N/A':
            return poster_url.replace('http://', 'https://', 1)
    except (requests.RequestException, ValueError):
        pass
    return POSTER_FALLBACK

def recommend_movies(movie_title, num_recommendations=5):
    idx = get_movie_index(movie_title)
    if idx is None:
        return []
    similarities = cosine_similarity(count_matrix[idx], count_matrix).flatten()
    sim_scores = list(enumerate(similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    recommended = []
    for i, score in sim_scores:
        title = movies.iloc[i]['title']
        year = None
        # If your dataset has a release_date column, extract year
        if 'release_date' in movies.columns:
            try:
                year = int(str(movies.iloc[i]['release_date'])[:4])
            except:
                year = None
        poster_url = fetch_poster_omdb(title, year)
        genres = str(movies.iloc[i].get('genres', '')).replace(' ', ' / ')
        recommended.append({
            'title': title,
            'poster': poster_url,
            'year': year,
            'genres': genres,
            'score': round(float(score) * 100),
        })
    return recommended

@app.get('/')
def home():
    poster_status = 'OMDb key configured' if OMDB_API_KEY else 'Add OMDB_API_KEY for posters'
    return render_template('index.html', movies=movies['title'].tolist(), poster_status=poster_status)


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'poster_provider': 'omdb-configured' if OMDB_API_KEY else 'fallback'})


@app.get('/api/recommendations')
def recommendations():
    title = request.args.get('title', '')
    try:
        limit = max(1, min(int(request.args.get('limit', 5)), 10))
    except ValueError:
        limit = 5
    if not title:
        return jsonify({'error': 'A movie title is required.'}), 400
    results = recommend_movies(title, limit)
    if not results:
        return jsonify({'error': 'Movie not found.'}), 404
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
