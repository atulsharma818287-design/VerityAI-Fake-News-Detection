# Verity AI 2.0 — News Intelligence & Classification

## Overview
A Python application that classifies news text with a machine learning model and can be extended with evidence-based verification using RAG.

## Features
- News text classification
- TF-IDF feature extraction
- Logistic Regression model
- Model evaluation using accuracy, precision, recall, and F1-score
- Multi-page Streamlit interface with analysis, session history, and project information
- Model-score display with clear limitations and evidence-verification status
- Automatically refreshed multi-topic live news radar with duplicate
  consolidation, search and filters
- Publisher thumbnails, source metadata, and current-feed screening charts
- SQLite-backed registration and password verification helpers in `auth.py`
- Planned RAG-based evidence retrieval

## Technology stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- feedparser

## Setup
1. Create and activate a virtual environment (PowerShell on Windows):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install the dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Place a labeled dataset at `data/news.csv`.
4. Train the model:
   ```bash
   python src/train.py
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

The Live News Feed page can monitor multiple topics at once. It refreshes RSS
feeds automatically every five minutes while the page is open, and also offers
a manual refresh button. It uses BBC World RSS for World News (including
publisher thumbnails) and India-focused Google News RSS for other selected
topics. Results are cached for five minutes, duplicate headlines are
consolidated across selected topics, and each is classified using the existing
scikit-learn model on its headline plus available RSS summary; the full article
is not downloaded.
Search, topic, model-label, and publication-date filters plus charts summarize
the currently fetched feed only; historical trend storage is not implemented.

## Dataset
This project expects a CSV file with at least these columns:
- `text`
- `label`

The expected labels are `REAL` and `FAKE`.

The checked-in CSV contains only a small illustrative sample, not a verified
or representative news dataset. Replace it with a suitably sourced and
documented dataset before drawing conclusions from model results.

## Authentication
`auth.py` provides SQLite user registration and password verification using
bcrypt password hashes. The database is stored locally as `users.db` and
excluded from Git. These helpers are not yet connected to the Streamlit UI.

## Evaluation
After training, the script reports accuracy, precision, recall, F1-score, and
a confusion matrix for a stratified held-out split. Results on a small or
unrepresentative dataset do not indicate real-world performance.

## Limitations
The model learns patterns from the training data and may misclassify
unfamiliar or misleading claims. Its predicted label and class scores are not
proof of factual accuracy. Live-feed results classify headlines and RSS
summaries, not complete articles. Feed thumbnails are supplied by the publisher.
The application does not retrieve independent evidence or perform RAG-based
verification; it is a text classifier, not a fact-checker.

## Future work
- Claim extraction
- Evidence retrieval
- RAG-based verification
- Source citation and explainability
- Multi-language support
