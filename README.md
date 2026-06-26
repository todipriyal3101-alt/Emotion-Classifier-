# Emotion Classifier — NLP Project

A 6-class emotion classifier (sadness, anger, love, surprise, fear, joy) trained
with Logistic Regression + TF-IDF (86.3% test accuracy), served via a FastAPI
backend and a clean web UI.

## Project structure

```
sentiment-app/
├── index.html              # Frontend UI ("Try it" widget)
├── backend/
│   ├── train.py             # Reproduces the notebook's cleaning + training
│   ├── app.py                # FastAPI server with /predict endpoint
│   └── requirements.txt
└── README.md
```

## Step 1 — Model is already trained ✅

Your `train.txt` was used to train the model already — the `backend/model/`
folder in this download already contains:
- `tfidf_vectorizer.pkl`
- `logistic_model.pkl`
- `label_map.pkl`

Test accuracy: **86.28%** (matches your notebook). You do NOT need to run
`train.py` again unless you want to retrain on a different/updated dataset.

(If you ever do need to retrain: put a `train.txt` in `backend/` and run
`python train.py` — it will overwrite the files in `model/`.)

## Step 2 — Run the API locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Visit `http://localhost:8000/docs` to test the `/predict` endpoint directly.

Example request:
```json
POST /predict
{ "text": "I am happy today" }
```

Example response:
```json
{ "label": "joy", "confidence": 0.91 }
```

## Step 3 — Connect the UI

Open `index.html` in your editor and change:

```js
const API_BASE_URL = "http://localhost:8000";
const DEMO_MODE = false;   // turn off the fake demo classifier
```

Then open `index.html` in your browser and try it.

## Step 4 — Push to GitHub

```bash
git init
git add .
git commit -m "Emotion classifier with UI + API"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Note: don't commit your raw `train.txt` or huge `.pkl` files if they're large —
add them to `.gitignore` and instead document how to regenerate them, or use
Git LFS if you want the trained model in the repo.

## Step 5 — Make it usable by others (not just visible)

GitHub Pages can host `index.html`, but it can't run Python — so your FastAPI
backend needs to live somewhere with a public URL. Free options:

- **Render** (render.com) — deploy `backend/` as a Web Service
- **Railway** (railway.app)
- **Hugging Face Spaces** — good fit for ML demos specifically

Once deployed, copy the public URL it gives you (e.g. `https://your-app.onrender.com`)
and update `API_BASE_URL` in `index.html` to that address, then push the change.

After that, anyone with your GitHub Pages link can type a sentence and get a
real prediction from your trained model — not just see a static page.

## Notes

- CORS is currently set to allow all origins (`*`) in `app.py` for easy testing.
  Restrict this to your actual frontend domain before calling it "production".
- Only the Logistic Regression model is currently wired into `/predict`. The
  dropdown for Naive Bayes variants is in the UI but not yet connected — let me
  know if you want all three models saved and selectable.
