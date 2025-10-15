import pandas as pd
from model import get_dominant_emotion
from service import get_songs_from_service

# Load local dataset
songs_df = pd.read_csv(r"C:\Users\HP\Downloads\NLP\NLP\spotify_plus_youtube.csv")
songs_df.columns = songs_df.columns.str.lower()

# Define emotion to genre mapping
emotion_to_genre = {
    "joy": ["pop", "dance", "electronic"],
    "sadness": ["acoustic", "soft rock", "ballad"],
    "anger": ["rock", "metal", "rap"],
    "fear": ["dark ambient", "classical", "lofi"],
    "surprise": ["indie", "experimental", "jazz"],
    "disgust": ["punk", "grunge"],
    "love": ["r&b", "soul", "romantic pop"],
    "neutral": ["chill", "instrumental", "lofi"]
}

def get_songs_for_emotion(emotion, n=10):
    emotion = emotion.lower()
    genres = emotion_to_genre.get(emotion, ["pop"])
    result = pd.DataFrame()

    if 'playlist_genre' not in songs_df.columns:
        raise ValueError("❌ 'playlist_genre' column not found in the dataset.")

    mask = songs_df['playlist_genre'].astype(str).str.lower().apply(
        lambda g: any(genre in g for genre in genres)
    )
    filtered = songs_df[mask]

    if not filtered.empty:
        selected = filtered.sample(n=min(n, len(filtered)), random_state=42)
        columns_to_show = [col for col in ['track_album_name', 'track_artist', 'playlist_genre', 'youtube_link'] if
                           col in filtered.columns]
        result = selected[columns_to_show].reset_index(drop=True)
    else:
        result = get_songs_from_service(emotion, genres, n)

    if result.empty:
        print(f"\n😔 Could not find any songs for '{emotion.title()}' from any source.")
        return result

    print(f"\n🎵 Top {len(result)} Recommended Songs for '{emotion.title()}' emotion:\n" + "-" * 65)
    for idx, row in result.iterrows():
        print(
            f"{idx + 1}. {row.get('track_album_name', 'Unknown Title')}  🎤 {row.get('track_artist', 'Unknown Artist')}")
        print(f"   Genre: {row.get('playlist_genre', 'N/A')}")
        print(f"   🔗 Link: {row.get('youtube_link', 'No link available')}\n")
    print("-" * 65)

    return result

text = input("Enter your prompt: ")
detected_emotion = get_dominant_emotion(text)
print("\nDetected Emotion →", detected_emotion.upper())

songs = get_songs_for_emotion(detected_emotion)