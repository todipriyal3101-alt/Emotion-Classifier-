"""
app.py — FastAPI backend that loads the trained model and serves predictions
to the index.html UI.

USAGE:
    1. Run train.py first (it creates the model/ folder).
    2. pip install -r requirements.txt
    3. uvicorn app:app --reload --port 8000
    4. Test at http://localhost:8000/docs
"""

import os
import string

import joblib
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

app = FastAPI(title="Emotion Classifier API")

# Allow requests from any origin (so GitHub Pages / other domains can call this).
# Tighten this to your actual frontend domain before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.pkl"))
label_map = joblib.load(os.path.join(MODEL_DIR, "label_map.pkl"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(ch for ch in text if not ch.isdigit())
    text = "".join(ch for ch in text if ch.isascii())
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


class PredictRequest(BaseModel):
    text: str
    model: str | None = None  # accepted but currently only one model is wired up


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Emotion classifier API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned = clean_text(req.text)
    vec = vectorizer.transform([cleaned])

    pred_id = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = float(max(probs))
    label = label_map[pred_id]

    return PredictResponse(label=label, confidence=confidence)
