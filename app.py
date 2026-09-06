from flask import Flask, jsonify, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
from dotenv import load_dotenv

load_dotenv()
OMDB_API_KEY = os.getenv('OMDB_API_KEY')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

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

def fetch_poster_omdb(title, year=None):
    """
    Fetch poster URL from OMDb API using movie title (and optionally year).
    """
    if not OMDB_API_KEY:
        return "https://via.placeholder.com/300x450?text=No+Poster"
    params = {
        't': title,
        'apikey': OMDB_API_KEY
    }
    if year:
        params['y'] = str(year)
    response = requests.get('http://www.omdbapi.com/', params=params, timeout=5)
    if response.status_code == 200:
        data = response.json()
        poster_url = data.get('Poster')
        if poster_url and poster_url != "N/A":
            return poster_url
    return "https://via.placeholder.com/300x450?text=No+Poster"

def recommend_movies(movie_title, num_recommendations=5):
    idx = get_movie_index(movie_title)
    if idx is None:
        return []
    similarities = linear_kernel(count_matrix[idx], count_matrix).flatten()
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
        recommended.append((title, poster_url))
    return recommended

@app.get('/')
def home():
    return render_template('index.html', movies=movies['title'].tolist())


@app.get('/health')
def health():
    return jsonify({'status': 'ok'})


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
    return jsonify([{'title': title, 'poster': poster} for title, poster in results])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
