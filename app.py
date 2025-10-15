import streamlit as st
import pandas as pd
from extract import get_dominant_emotion, songs_df, get_songs_for_emotion

st.set_page_config(page_title="Emotion-Based Playlist Generator", page_icon="🎵", layout="wide")

st.title("🎵 Emotion-Based Playlist Generator")
st.write("Enter a prompt or text, and we'll detect your emotion and recommend a playlist!")

user_text = st.text_area("How do you feel Today : ? What's the mood?", "", height=100)

if st.button("Generate Playlist"):
    if not user_text.strip():
        st.warning("Please enter some text to generate a playlist.")
    else:
        # Detect emotion
        emotion = get_dominant_emotion(user_text)
        st.success(f"Detected Emotion: **{emotion.upper()}**")

        # Get songs
        playlist = get_songs_for_emotion(emotion, n=10)

        if not playlist.empty:
            st.subheader("Recommended Songs 🎶")
            for idx, row in playlist.iterrows():
                st.markdown(
                    f"**{idx + 1}. {row.get('track_album_name', 'Unknown Title')}**  🎤 {row.get('track_artist', 'Unknown Artist')}")
                st.markdown(f"Genre: {row.get('playlist_genre', 'N/A')}")
                st.markdown(f"[YouTube Link]({row.get('youtube_link', '#')})")
                st.markdown("---")
        else:
            st.info("No songs found for this emotion.")
