# Emotion Classifier — NLP Project

🔗 Live demo: https://todipriyal3101-alt.github.io/Emotion-Classifier-/

A 6-class emotion classifier (sadness, anger, love, surprise, fear, joy) trained with Logistic Regression + TF-IDF (86.3% test accuracy), served via a FastAPI backend and a clean web UI. Click the link above to try it instantly — no setup needed.

## Project structure

    sentiment-app/
    ├── index.html              # Frontend UI ("Try it" widget)
    ├── backend/
    │   ├── train.py             # Reproduces the notebook's cleaning + training
    │   ├── app.py                # FastAPI server with /predict endpoint
    │   ├── requirements.txt
    │   └── model/                 # Trained model files
    └── README.md

## How it's deployed

- Frontend: hosted on GitHub Pages, served directly from this repo's main branch
- Backend: hosted on Render as a live API at https://emotion-classifier-api.onrender.com

Because both pieces are deployed, the link at the top of this README works for anyone — it's not just a local demo.

Note: the backend runs on Render's free tier, which "sleeps" after periods of inactivity. If the first request after a while takes 30-50 seconds, that's expected — it's just waking back up.

## Running it locally (optional)

You don't need to do this to use the live demo, but if you want to run everything on your own machine:

1. Install dependencies and start the API

    cd backend
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Visit http://localhost:8000/docs to test the /predict endpoint directly.

Example request:

    POST /predict
    { "text": "I am happy today" }

Example response:

    { "label": "joy", "confidence": 0.91 }

2. Point the UI at your local API

Open index.html and change:

    const API_BASE_URL = "http://localhost:8000";
    const DEMO_MODE = false;

Then open index.html in your browser and try it.

## Retraining the model (optional)

The model is already trained and included in backend/model/. If you want to retrain it on new data:

1. Put a train.txt file inside backend/
2. Run: python train.py
3. This regenerates the files inside backend/model/

## Notes

- The model isn't perfect — it's about 86% accurate, so it sometimes gets confused on subtle or mixed emotions.
- CORS is currently open to all origins in app.py, which is fine for a demo but should be locked down before treating this as a "production" app.
- Only the Logistic Regression model is currently wired into /predict. The dropdown for Naive Bayes variants is shown in the UI but not yet connected.
