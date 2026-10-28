import re

import pandas as pd

from src.news_feed import fetch_news as _fetch_news
from src.predict import load_model, predict_news


def fetch_news(
    topics: str | list[str] = "India",
    limit_per_topic: int = 15,
) -> list[dict[str, str]]:
    """Fetch topic feeds and remove repeated headlines across those feeds."""
    selected_topics = [topics] if isinstance(topics, str) else topics
    if not selected_topics:
        raise ValueError("Select at least one news topic.")

    articles_by_headline: dict[str, dict[str, str]] = {}
    for topic in selected_topics:
        for item in _fetch_news(topic, limit=limit_per_topic):
            source = item["source"]
            headline = item["headline"]
            canonical_headline = re.sub(
                rf"\s[-–|]\s{re.escape(source)}$",
                "",
                headline,
                flags=re.IGNORECASE,
            )
            headline_key = re.sub(
                r"[^a-z0-9]+", " ", canonical_headline.casefold()
            ).strip()
            if not headline_key:
                continue

            existing = articles_by_headline.get(headline_key)
            if existing is not None:
                categories = existing["topic"].split(", ")
                if topic not in categories:
                    categories.append(topic)
                    existing["topic"] = ", ".join(categories)
                continue

            articles_by_headline[headline_key] = {
                "headline": headline,
                "link": item["url"],
                "published": item["published_at"],
                "summary": item["summary"],
                "source": source,
                "image_url": item.get("image_url", ""),
                "fetched_at": item["fetched_at"],
                "topic": topic,
            }

    return list(articles_by_headline.values())


def detect_news(
    articles: list[dict[str, str]],
    model=None,
) -> pd.DataFrame:
    """Classify each feed item's headline and available summary."""
    if model is None:
        model = load_model()

    results: list[dict[str, object]] = []
    for article in articles:
        headline = article.get("headline", "").strip()
        summary = article.get("summary", "").strip()
        text = f"{headline} {summary}".strip()
        if not text:
            continue

        prediction = predict_news(text, model)
        results.append(
            {
                **article,
                "image_url": article.get("image_url", ""),
                "topic": article.get("topic", "Uncategorized"),
                "prediction": prediction["prediction"],
                "scores": prediction["class_scores"],
            }
        )

    dataframe = pd.DataFrame(results)
    if not dataframe.empty:
        published_at = pd.to_datetime(
            dataframe["published"],
            utc=True,
            errors="coerce",
        )
        dataframe = (
            dataframe.assign(_published_sort=published_at)
            .sort_values("_published_sort", ascending=False, na_position="last")
            .drop(columns="_published_sort")
            .reset_index(drop=True)
        )
    return dataframe
