import os
import json
import pandas as pd
import google.generativeai as genai


try:
    api_key = "AIzaSyBMNKKsznIyqjoBiMcT1V3COSBaQ-eVAn0"
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(e)

    pass


def get_songs_from_service(emotion, genres, n=10):



    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    Recommend {n} songs for the emotion '{emotion}' that fit these genres: {', '.join(genres)}.
    Please provide the response as a valid JSON array where each object has the following keys:
    - "track_album_name": The name of the song.
    - "track_artist": The name of the artist.
    - "playlist_genre": One of the suggested genres.
    - "youtube_link": A valid YouTube search link for the song.

    Example format:
    [
      {{
        "track_album_name": "Bohemian Rhapsody",
        "track_artist": "Queen",
        "playlist_genre": "rock",
        "youtube_link": "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
      }}
    ]
    """

    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's valid JSON
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        songs_data = json.loads(cleaned_text)
        return pd.DataFrame(songs_data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"❌ Error fetching or parsing data from Gemini: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure