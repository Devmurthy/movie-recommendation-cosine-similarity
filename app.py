import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Set page config as the very first Streamlit command!
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Load OMDb API Key from .env file
load_dotenv()
OMDB_API_KEY = os.getenv('OMDB_API_KEY')

# 1. Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('movies.csv')
    # Combine relevant features into a single string
    features = ['genres', 'keywords', 'cast', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')
    df['combined_features'] = df.apply(lambda row: ' '.join([str(row[feat]) for feat in features]), axis=1)
    return df

movies = load_data()

# 2. Vectorize text features and compute cosine similarity
@st.cache_data
def compute_similarity(df):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(count_matrix)
    return cosine_sim

cosine_sim = compute_similarity(movies)

# 3. Helper functions
def get_movie_index(title):
    result = movies[movies['title'].str.lower() == title.lower()]
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
    response = requests.get('http://www.omdbapi.com/', params=params)
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
    sim_scores = list(enumerate(cosine_sim[idx]))
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

# 4. Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Get movie recommendations based on your favorite movie using content-based filtering and cosine similarity.")

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie you like:", movie_list)
num_recs = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Show Recommendations"):
    recommendations = recommend_movies(selected_movie, num_recs)
    if recommendations:
        cols = st.columns(num_recs)
        for idx, (title, poster_url) in enumerate(recommendations):
            with cols[idx]:
                st.image(poster_url, width=150)
                st.caption(title)
    else:
        st.warning("Movie not found in the database.")

st.markdown("---")
st.markdown("**Project by N K L N MURTHY**")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

/* Main background and font */
html, body, .stApp {
    background: #f4f6fa !important;
    font-family: 'Montserrat', sans-serif !important;
    color: #111 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Montserrat', sans-serif !important;
    color: #009688 !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}

/* Labels and captions */
label, .movie-caption, .stMarkdown, .st-bb, .st-cq {
    color: #222 !important;
    font-weight: 500 !important;
}

/* Movie card styling */
.movie-card {
    background: #fff !important;
    border-radius: 12px;
    box-shadow: 0 2px 12px 0 rgba(0,0,0,0.07);
    padding: 1rem 0.5rem;
    margin: 0.5rem 0;
    transition: transform 0.2s, box-shadow 0.2s;
}
.movie-card:hover {
    transform: translateY(-4px) scale(1.03);
    box-shadow: 0 8px 24px 0 rgba(0,150,136,0.10);
}
.movie-card img {
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
.movie-caption {
    text-align: center;
    font-size: 1.05rem;
    color: #222 !important;
}

/* Button styling */
.stButton>button {
    background: #009688;
    color: #fff;
    border: none;
    border-radius: 20px;
    padding: 12px 28px;
    font-size: 1rem;
    font-weight: 700;
    font-family: 'Montserrat', sans-serif;
    box-shadow: 0 2px 8px 0 rgba(0,150,136,0.10);
    transition: background 0.2s, transform 0.2s;
}
.stButton>button:hover {
    background: #00796b;
    transform: translateY(-2px);
}

/* Selectbox and slider label color */
.css-1cpxqw2, .css-1v0mbdj, .css-1kyxreq, .css-16huue1, .css-1offfwp, .css-1q8dd3e {
    color: #222 !important;
}

/* Make selectbox and text input text black on white */
.stSelectbox input, .stTextInput input {
    color: #111 !important;
    background: #fff !important;
    border-radius: 8px !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #009688;
    border-radius: 4px;
}
::-webkit-scrollbar-track {
    background: #f4f6fa;
}
</style>
""", unsafe_allow_html=True)
