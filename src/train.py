from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "news.csv"
MODEL_PATH = ROOT / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(["FAKE", "REAL"])]

    if df.empty:
        raise ValueError("Dataset contains no usable rows with FAKE or REAL labels.")

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both FAKE and REAL labels.")

    label_counts = df["label"].value_counts()
    if label_counts.min() < 2:
        raise ValueError(
            "Each label must have at least 2 examples for a meaningful train/test split."
        )

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_test, y_pred, labels=["FAKE", "REAL"], target_names=["FAKE", "REAL"]
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model.named_steps["tfidf"], VECTORIZER_PATH)


if __name__ == "__main__":
    main()
