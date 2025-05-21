import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {"not", "no", "nor"}

MAX_LEN = 200

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load model
with open('tokenizer_finetuned.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model('best_lstm_model_finetuned.h5')

def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    label = "Positive" if pred >= 0.5 else "Negative"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, round(confidence, 2)
