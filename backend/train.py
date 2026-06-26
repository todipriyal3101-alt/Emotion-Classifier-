"""
train.py — reproduces the cleaning + training pipeline from NLP_1.ipynb
and saves the trained model artifacts to disk so the API can load them.

USAGE:
    1. Put your train.txt in the same folder as this script.
    2. pip install -r requirements.txt
    3. python train.py
    4. This creates a `model/` folder with:
         - tfidf_vectorizer.pkl
         - logistic_model.pkl
         - label_map.pkl   (e.g. {0: "sadness", 1: "anger", ...})
"""

import os
import re
import string
import joblib
import pandas as pd

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Same cleaning steps as the notebook: lowercase, strip punctuation,
    strip digits, strip non-ascii (emojis), remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(ch for ch in text if not ch.isdigit())
    text = "".join(ch for ch in text if ch.isascii())
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def main():
    print("Loading train.txt ...")
    df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

    # Build label <-> id mapping in order of first appearance (same as notebook)
    unique_emotions = df["emotion"].unique()
    label_to_id = {label: i for i, label in enumerate(unique_emotions)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    df["label_id"] = df["emotion"].map(label_to_id)

    print("Cleaning text ...")
    df["text"] = df["text"].apply(clean_text)

    x = df["text"]
    y = df["label_id"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print("Vectorizing (TF-IDF) ...")
    vectorizer = TfidfVectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    print("Training Logistic Regression ...")
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_tfidf, y_train)

    preds = model.predict(x_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
    joblib.dump(model, "model/logistic_model.pkl")
    joblib.dump(id_to_label, "model/label_map.pkl")

    print("\nSaved to model/:")
    print("  - tfidf_vectorizer.pkl")
    print("  - logistic_model.pkl")
    print("  - label_map.pkl  ->", id_to_label)


if __name__ == "__main__":
    main()
