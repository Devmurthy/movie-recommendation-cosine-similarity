# Movie Recommendation System (Cosine Similarity)

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?logo=streamlit)](https://devmurthy-movie-recommendation-cosine-similarity-app-xuqzvc.streamlit.app)

**Live Demo:** [Click here to try the app!](https://devmurthy-movie-recommendation-cosine-similarity-app-xuqzvc.streamlit.app)

---

## Overview
This is a Movie Recommendation System built with Flask. It recommends movies based on your favorite movie using content-based filtering and cosine similarity. Movie posters are fetched from the OMDb API.

## Features
- Select a movie you like and get similar recommendations
- Beautiful, modern UI
- Movie posters via OMDb API

## Setup
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OMDb API key:
   ```env
   OMDB_API_KEY=your_actual_omdb_api_key_here
   ```
4. Run the app:
   ```bash
   python app.py
   ```

## Deploy on Render

Create a Render **Web Service** connected to this repository with:

- **Build Command:** `python -m pip install -r requirements.txt`
- **Start Command:** `python -m gunicorn --bind 0.0.0.0:$PORT app:app`
- **Environment Variable:** `OMDB_API_KEY` (optional, for movie posters)

The service exposes `/health` for a health check. On Render's free tier, services can still spin down after inactivity; use a paid instance if the site must remain continuously warm.

If the Render service already exists, update these two commands in its **Settings** and trigger a manual redeploy. A `render.yaml` file is only applied automatically when creating or syncing a Render Blueprint.

---

**Project by N K L N MURTHY** 