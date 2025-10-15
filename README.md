# 🎵 Emotion-Based Playlist Generator

A Natural Language Processing (NLP) application that analyzes text to detect emotions and generates personalized music playlists based on the detected emotional state.

## 🌟 Features

- **Emotion Detection**: Uses a fine-tuned transformer model to analyze text and detect 27 different emotions
- **Playlist Generation**: Recommends songs based on detected emotions using local dataset and AI-powered suggestions
- **Web Interface**: Interactive Streamlit application for easy user interaction
- **Fallback System**: Integrates with Google Gemini AI for additional song recommendations when local dataset is insufficient
- **Multiple Data Sources**: Combines Spotify dataset with YouTube links for comprehensive music recommendations

## 🏗️ Project Structure

```
├── app.py                      # Main Streamlit web application
├── model.py                    # Emotion detection model and inference
├── train.py                    # Model training script for emotion classification
├── extract.py                  # Song extraction and emotion-to-genre mapping
├── service.py                  # External API service for additional song recommendations
├── spotify_plus_youtube.csv    # Local music dataset with Spotify and YouTube data
└── README.md                   # Project documentation
```

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: PyTorch, Transformers (Hugging Face)
- **NLP Model**: Fine-tuned RoBERTa for emotion classification
- **AI Integration**: Google Gemini API
- **Data Processing**: Pandas, NumPy
- **Dataset**: GoEmotions (for training), Spotify + YouTube dataset (for recommendations)

## 🎯 Supported Emotions

The model detects 27 fine-grained emotions and maps them to 8 general emotional categories:

### Fine-grained emotions:
`admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`

### General emotion mapping:
- **Joy**: pop, dance, electronic
- **Sadness**: acoustic, soft rock, ballad
- **Anger**: rock, metal, rap
- **Fear**: dark ambient, classical, lofi
- **Love**: r&b, soul, romantic pop
- **Surprise**: indie, experimental, jazz
- **Disgust**: punk, grunge
- **Neutral**: chill, instrumental, lofi

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emotion-playlist-generator
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas torch transformers scikit-learn google-generativeai datasets
   ```

3. **Download the pre-trained model**
   - Ensure the emotion model is available at the path specified in `model.py`
   - Default path: `C:\Users\HP\Downloads\NLP\NLP\emotion_model`

4. **Set up the dataset**
   - Place the `spotify_plus_youtube.csv` file in the project directory
   - Update the dataset path in `extract.py` if needed

5. **Configure API keys** (Optional)
   - Add your Google API key in `service.py` for enhanced recommendations

## 💻 Usage

### Running the Web Application

```bash
streamlit run app.py
```

1. Open your browser and navigate to the local Streamlit URL (typically `http://localhost:8501`)
2. Enter text describing your current mood or feelings
3. Click "Generate Playlist" to get emotion-based music recommendations
4. Enjoy your personalized playlist with YouTube links!

### Training Your Own Model

```bash
python train.py --model_name roberta-base --output_dir ./my_emotion_model --num_epochs 3
```

**Training arguments:**
- `--model_name`: Base transformer model (default: roberta-base)
- `--output_dir`: Directory to save trained model (default: ./emotion_model_trained)
- `--num_epochs`: Number of training epochs (default: 3)
- `--train_batch_size`: Training batch size (default: 32)
- `--eval_batch_size`: Evaluation batch size (default: 64)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--seed`: Random seed for reproducibility (default: 42)

## 📊 Model Performance

The emotion detection model is trained on the GoEmotions dataset and evaluated using:
- **F1 Score** (Micro & Macro averaged)
- **Precision** (Micro averaged)
- **Recall** (Micro averaged)

## 🎨 Example Usage

**Input:** "I'm feeling really excited about my new job! Can't wait to start this amazing journey."

**Output:**
- **Detected Emotion:** JOY
- **Recommended Genre:** Pop, Dance, Electronic
- **Sample Songs:** Upbeat pop songs with high energy and positive vibes

## 🔧 Configuration

### Model Configuration
- Update the `MODEL_PATH` in `model.py` to point to your trained model
- Modify emotion-to-genre mapping in `extract.py` based on your preferences

### Dataset Configuration
- Update dataset paths in `extract.py` and `service.py`
- Customize the CSV column names if using a different dataset format

## 🚦 Error Handling

The application includes robust error handling for:
- Missing model files
- Invalid dataset formats
- API connection issues
- Empty or invalid user inputs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Dependencies

- `streamlit`: Web application framework
- `torch`: Deep learning framework
- `transformers`: Hugging Face transformers library
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning metrics
- `google-generativeai`: Google Gemini AI integration
- `datasets`: Dataset loading and processing

## 📧 Contact

For questions, suggestions, or support, please create an issue in the repository.

---

**Happy listening! 🎵✨**