from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer from the local directory
# Change the path to where your model is actually stored
MODEL_PATH = r"C:\Users\HP\Downloads\NLP\NLP\emotion_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# You can define your label mapping manually or from model config
# Example: labels order = ['joy', 'sadness', 'anger', 'fear', 'neutral', ...]
#label_mapping = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise', 'disgust', 'neutral']
label_mapping=["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise"]
general_mapping = {
    "admiration": "love",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "love",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "love",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "love",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "neutral",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise"
}
def get_dominant_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    return general_mapping[label_mapping[pred_label]]
