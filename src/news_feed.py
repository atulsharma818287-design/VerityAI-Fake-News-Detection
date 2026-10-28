import re
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import feedparser

FEED_URL = "https://news.google.com/rss/search"
WORLD_FEED_URL = "https://feeds.bbci.co.uk/news/world/rss.xml"
USER_AGENT = "VerityAI/1.0 (news feed reader)"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(unescape(value))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def fetch_news(query: str = "India", limit: int = 30) -> list[dict[str, str]]:
    query = query.strip()
    if not query:
        raise ValueError("Enter a topic or keyword to search for news.")
    if limit < 1:
        raise ValueError("The result limit must be at least 1.")

    is_world_feed = query.casefold() in {"world", "world news"}
    if is_world_feed:
        url = WORLD_FEED_URL
    else:
        params = urlencode(
            {
                "q": query,
                "hl": "en-IN",
                "gl": "IN",
                "ceid": "IN:en",
            }
        )
        url = f"{FEED_URL}?{params}"

    request = Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )

    try:
        with urlopen(request, timeout=15) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not fetch the news feed: {exc}") from exc

    feed = feedparser.parse(payload)
    if feed.bozo and not feed.entries:
        raise RuntimeError("The news provider returned an invalid RSS feed.")

    fetched_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    articles: list[dict[str, str]] = []

    for entry in feed.entries:
        title = _plain_text(entry.get("title", ""))
        url = entry.get("link", "").strip()
        parsed_url = urlparse(url)
        if not title or parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            continue

        source = entry.get("source", {}).get(
            "title",
            "BBC News" if is_world_feed else "Google News",
        )
        image_url = ""
        for image in entry.get("media_thumbnail", []) + entry.get(
            "media_content", []
        ):
            candidate = image.get("url", "").strip()
            parsed_image_url = urlparse(candidate)
            if (
                parsed_image_url.scheme in {"http", "https"}
                and parsed_image_url.netloc
            ):
                image_url = candidate
                break

        articles.append(
            {
                "headline": title,
                "summary": _plain_text(entry.get("summary", "")),
                "source": source,
                "url": url,
                "image_url": image_url,
                "published_at": entry.get("published", "Unknown"),
                "fetched_at": fetched_at,
            }
        )
        if len(articles) >= limit:
            break

    return articles
