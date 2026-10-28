import unittest
from unittest.mock import patch

from src.live_news import detect_news, fetch_news


def article(
    headline: str,
    published_at: str = "Tue, 27 Oct 2026 10:00:00 GMT",
    summary: str = "A short feed summary.",
) -> dict[str, str]:
    url = f"https://news.example/{headline.lower().replace(' ', '-')}"
    return {
        "headline": headline,
        "url": url,
        "link": url,
        "published_at": published_at,
        "published": published_at,
        "summary": summary,
        "source": "Example News",
        "image_url": "",
        "fetched_at": "2026-10-27 10:00:00 UTC",
        "topic": "World News",
    }


class LiveNewsTests(unittest.TestCase):
    def test_fetch_news_consolidates_duplicate_headlines_across_topics(self) -> None:
        original = article("Major policy update for India")
        repeated = article("Major policy update for India - Example News")

        with patch(
            "src.live_news._fetch_news",
            side_effect=[[original], [repeated]],
        ) as fetch_feed:
            results = fetch_news(["India", "Politics"])

        self.assertEqual(fetch_feed.call_count, 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["headline"], original["headline"])
        self.assertEqual(results[0]["topic"], "India, Politics")

    def test_detect_news_uses_existing_predictor_on_headline_and_summary(self) -> None:
        story = article("Headline text")
        model = object()

        with patch(
            "src.live_news.predict_news",
            return_value={
                "prediction": "REAL",
                "class_scores": {"FAKE": 0.2, "REAL": 0.8},
            },
        ) as predictor:
            results = detect_news([story], model)

        predictor.assert_called_once_with("Headline text A short feed summary.", model)
        self.assertEqual(results.loc[0, "prediction"], "REAL")
        self.assertEqual(results.loc[0, "scores"], {"FAKE": 0.2, "REAL": 0.8})
        self.assertEqual(results.loc[0, "link"], story["url"])


if __name__ == "__main__":
    unittest.main()
