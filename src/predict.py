from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fake_news_model.pkl"


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it with: python src/train.py"
        )
    return joblib.load(MODEL_PATH)


def predict_news(text: str, model=None) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("Please enter a news article.")

    if model is None:
        model = load_model()

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    scores = {
        label: float(probability)
        for label, probability in zip(classes, probabilities)
    }
    return {"prediction": prediction, "class_scores": scores}


if __name__ == "__main__":
    article = input("Enter a news article: ")
    result = predict_news(article, load_model())
    print("Predicted label:", result["prediction"])
    print("Class scores:", result["class_scores"])
