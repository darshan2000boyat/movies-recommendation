from fastapi import FastAPI, Query, HTTPException
import pickle
import pandas as pd
import httpx
from io import BytesIO
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PICKLE_URL = "https://storage.googleapis.com/learning_machine_learning/movie_recommendation_data.pkl"

app = FastAPI()
app.state.data_loaded = False
app.state.data = None

@app.on_event("startup")
async def load_data():
    """Load data asynchronously on startup with retry logic"""
    global movies, vectors, similarity
    
    logger.info("Starting data download...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(PICKLE_URL)
            response.raise_for_status()
            
            # Stream content in chunks
            content = b''
            for chunk in response.iter_bytes():
                content += chunk
                logger.info(f"Received {len(content)} bytes...")
            
            logger.info("Unpickling data...")
            data = pickle.load(BytesIO(content))
            
            # Store in app state
            app.state.movies = data["movies"]
            app.state.vectors = data["vectors"]
            app.state.similarity = data["similarity"]
            app.state.data_loaded = True
            
            logger.info("Data loaded successfully!")
            
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        app.state.data_loaded = False

def recommend(movie_name: str):
    """Get movie recommendations with data validation"""
    if not app.state.data_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    movies = app.state.movies
    similarity = app.state.similarity
    
    matches = movies[movies['title'].str.lower() == movie_name.lower()]
    if matches.empty:
        close_matches = movies[
            movies['title'].str.lower().str.contains(movie_name.lower())
        ].head(5)['title'].tolist()
        raise HTTPException(
            status_code=404,
            detail=f"Movie not found. Similar titles: {close_matches}"
        )
    
    movie_index = matches.index[0]
    distances = list(enumerate(similarity[movie_index]))
    sorted_distances = sorted(distances, reverse=True, key=lambda x: x[1])
    recommendations = [movies.iloc[i[0]].title for i in sorted_distances[1:6]]
    
    return recommendations

@app.get("/recommend")
async def get_recommendations(
    movie: str = Query(..., description="Movie title to get recommendations for")
):
    try:
        results = recommend(movie)
        return {
            "movie": movie,
            "recommendations": results
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail="Internal server error")