from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model as load_trained_model
from src.predict import predict_news
from src.live_news import detect_news, fetch_news

st.set_page_config(
    page_title="Verity AI 2.0 | News Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0b1020;
        --panel: #121a2d;
        --border: #25314b;
        --text: #edf2ff;
        --muted: #9ba9c5;
        --accent: #a5b4fc;
        --cyan: #67e8f9;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: "DM Sans", "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #0e1527;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3 {
        letter-spacing: -0.04em;
    }

    .hero {
        background: linear-gradient(120deg, #182547, #171a36 65%, #102b3c);
        border: 1px solid #344363;
        border-radius: 22px;
        padding: 30px;
        margin: 10px 0 24px;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .hero-title {
        color: white;
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 750;
        letter-spacing: -0.05em;
        line-height: 1.12;
        margin: 12px 0;
    }

    .hero-copy, .muted {
        color: #c0cce3;
        font-size: 0.95rem;
        line-height: 1.7;
        max-width: 700px;
    }

    .brand {
        font-size: 1.6rem;
        font-weight: 750;
        letter-spacing: -0.04em;
    }

    .brand span {
        color: var(--accent);
    }

    .result-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 22px;
        margin: 8px 0 18px;
    }

    .result-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 8px 0;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: 0;
        border-radius: 11px;
        min-height: 48px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        border: 1px solid #c4b5fd;
        filter: brightness(1.12);
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
    }

    textarea {
        border-radius: 12px !important;
    }

    hr {
        border-color: var(--border);
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 22px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return load_trained_model()


FEED_REFRESH_SECONDS = 300


@st.cache_data(ttl=FEED_REFRESH_SECONDS)
def load_live_news(topics: tuple[str, ...]):
    return fetch_news(list(topics))


def initialize_state():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("article", "")


def load_sample_claim():
    st.session_state.article = (
        "Fictional sample for testing: A city council announced a new public "
        "garden project that will open next summer."
    )


initialize_state()

with st.sidebar:
    st.markdown(
        '<div class="brand">◈ Verity<span>AI</span> 2.0</div>',
        unsafe_allow_html=True,
    )
    st.caption("NEWS INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Analyze News", "Live News Feed", "Analysis History", "About Project"],
        index=1,
    )

    st.divider()
    st.markdown("#### System status")
    st.markdown("🟢 Application interface ready")
    st.markdown(
        "🟢 ML model available"
        if (Path(__file__).resolve().parent / "models" / "fake_news_model.pkl").is_file()
        else "🟠 ML model not trained"
    )
    st.divider()
    st.markdown(
        '<p class="muted">AI-assisted text classification. '
        "Verify important claims with reliable sources.</p>",
        unsafe_allow_html=True,
    )

model = None
model_error = None

try:
    model = load_model()
except FileNotFoundError as exc:
    model_error = str(exc)


@st.fragment(run_every=FEED_REFRESH_SECONDS)
def refresh_live_news_automatically(
    topics: tuple[str, ...],
    force_refresh: bool = False,
) -> None:
    """Keep the live feed current while its page is open."""
    if st.session_state.pop("_feed_refresh_rerun_pending", False) and not force_refresh:
        return

    if not topics:
        if force_refresh:
            st.warning("Choose at least one news topic to refresh the radar.")
        return

    now = datetime.now().timestamp()
    last_topics = st.session_state.get("feed_topics")
    last_fetched = st.session_state.get("feed_fetched_epoch", 0.0)
    refresh_is_due = (
        force_refresh
        or topics != last_topics
        or now - last_fetched >= FEED_REFRESH_SECONDS
    )
    if not refresh_is_due:
        return

    if model is None:
        st.error(
            "The trained model could not be loaded. "
            "Run `python src/train.py` to create it."
        )
        if model_error:
            st.caption(model_error)
        return

    try:
        with st.spinner("Fetching headlines and analyzing news..."):
            if force_refresh:
                load_live_news.clear()
            articles = load_live_news(topics)
            if articles:
                st.session_state.feed_results = detect_news(articles, model)
            else:
                st.session_state.feed_results = pd.DataFrame()
            st.session_state.feed_topics = topics
            st.session_state.feed_fetched_at = datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
            st.session_state.feed_fetched_epoch = datetime.now().timestamp()
            st.session_state["_feed_refresh_rerun_pending"] = True
            st.rerun(scope="app")
    except (ValueError, RuntimeError) as exc:
        st.error(f"Could not process news: {exc}")


if page == "Live News Feed":
    st.title("Live news feed")
    st.markdown(
        '<p class="muted">Fetch recent headlines and classify them automatically '
        "with the trained model. Feed summaries may be available, but full article "
        "text is not fetched.</p>",
        unsafe_allow_html=True,
    )
    st.warning(
        "A FAKE or REAL label is only the model's text classification. "
        "It does not prove truth or falsehood. Evidence verification is not connected."
    )

    topics = st.multiselect(
        "Topics to monitor",
        [
            "World News",
            "India",
            "Technology",
            "Politics",
            "Business",
            "Artificial Intelligence",
            "Sports",
            "Health",
        ],
        default=["World News"],
    )
    refresh_column, status_column = st.columns([1, 2])
    refresh = refresh_column.button(
        "🔄 Refresh now",
        type="primary",
        use_container_width=True,
    )
    status_column.caption(
        f"Monitoring {len(topics)} topic{'s' if len(topics) != 1 else ''} · "
        "RSS feed refreshes automatically every 5 minutes while this page is open"
        if topics
        else "Choose at least one topic to monitor"
    )
    refresh_live_news_automatically(tuple(topics), force_refresh=refresh)

    results = st.session_state.get("feed_results")
    if results is not None and not results.empty:
        st.caption(
            f"Last refreshed: {st.session_state.get('feed_fetched_at', 'unknown')} "
            "· Duplicate headlines are consolidated across selected topics"
        )
        category_options = sorted(
            {
                category
                for categories in results["topic"]
                for category in categories.split(", ")
            }
        )
        filter_columns = st.columns([2, 1, 1, 1])
        search_term = filter_columns[0].text_input(
            "Search headlines and summaries",
            placeholder="Search this feed",
        ).strip().casefold()
        selected_categories = filter_columns[1].multiselect(
            "Filter topics",
            category_options,
            default=category_options,
        )
        selected_prediction = filter_columns[2].selectbox(
            "Model label",
            ["All", "FAKE", "REAL"],
        )
        date_range = filter_columns[3].selectbox(
            "Published",
            ["Any time", "Past 24 hours", "Past 7 days"],
        )

        filtered = results.copy()
        if selected_categories:
            filtered = filtered[
                filtered["topic"].map(
                    lambda value: any(
                        category in value.split(", ")
                        for category in selected_categories
                    )
                )
            ]
        else:
            filtered = filtered.iloc[0:0]
        if selected_prediction != "All":
            filtered = filtered[filtered["prediction"] == selected_prediction]
        if search_term:
            searchable = (
                filtered["headline"].fillna("")
                + " "
                + filtered["summary"].fillna("")
                + " "
                + filtered["source"].fillna("")
            ).str.casefold()
            filtered = filtered[searchable.str.contains(search_term, regex=False)]
        if date_range != "Any time":
            days = 1 if date_range == "Past 24 hours" else 7
            published_at = pd.to_datetime(
                filtered["published"],
                utc=True,
                errors="coerce",
            )
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
            filtered = filtered[published_at.isna() | (published_at >= cutoff)]

        if filtered.empty:
            st.info("No headlines match the selected filters.")

        metric_columns = st.columns(3)
        metric_columns[0].metric("Stories shown", len(filtered))
        metric_columns[1].metric("Sources represented", filtered["source"].nunique())
        metric_columns[2].metric(
            "Model flagged",
            int((filtered["prediction"] == "FAKE").sum()),
        )

        st.markdown(f"### Headlines ({len(filtered)})")
        for _, article in filtered.iterrows():
            with st.container(border=True):
                image_url = article.get("image_url", "")
                if image_url:
                    image_column, story_column = st.columns([1, 2])
                    with image_column:
                        st.image(
                            image_url,
                            caption=article["source"],
                            width="stretch",
                        )
                else:
                    story_column = st.container()

                with story_column:
                    st.subheader(article["headline"])
                    st.caption(f"Topic: {article['topic']}")

                    prediction = str(article["prediction"]).upper()
                    if prediction == "FAKE":
                        st.warning("Model prediction: Potentially misleading")
                    else:
                        st.info(
                            f"Model prediction: Classified as {prediction}"
                        )

                    st.caption(f"Source: {article['source']}")
                    st.caption(f"Published: {article['published'] or 'Unknown'}")
                    st.caption(f"Fetched: {article['fetched_at']}")
                    if article["summary"]:
                        with st.expander("Feed summary"):
                            st.write(article["summary"])
                    st.link_button("Read original source", article["link"])

        st.caption(
            "These are ML screening results, not verified facts. "
            "The model uses headline and available feed-summary text only. "
            "Check the original reporting before drawing conclusions."
        )
    elif results is not None and results.empty:
        st.info("No usable headlines were returned. Try another topic.")
    elif results is not None:
        st.info("No headlines match the selected filters.")

elif page == "Analyze News":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AI-powered news intelligence</div>
            <div class="hero-title">Know the facts.<br>Follow the evidence.</div>
            <div class="hero-copy">
                Explore how a machine-learning model classifies news text.
                This demo does not independently verify claims or retrieve evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    first_metric, second_metric, third_metric = st.columns(3)
    first_metric.metric("Analyses this session", len(st.session_state.history))
    second_metric.metric("ML model", "Ready" if model else "Unavailable")
    third_metric.metric("Evidence verification", "Not connected")

    st.subheader("Analyze an article")
    st.markdown(
        '<p class="muted">Paste the full article where possible. '
        "A headline alone may not provide enough context.</p>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        input_mode = st.radio(
            "Input type",
            ["Full article", "Headline / claim"],
            horizontal=True,
        )
        st.text_area(
            "News content",
            key="article",
            height=220,
            placeholder="Paste a headline, news claim, or complete article here...",
            label_visibility="collapsed",
        )
        st.caption(f"{len(st.session_state.article)} characters · {input_mode}")

        with st.expander("Try a sample input"):
            st.write("This fictional example is only for testing the interface.")
            st.button("Load sample claim", on_click=load_sample_claim)

        analyze = st.button(
            "🔎  Analyze news",
            type="primary",
            use_container_width=True,
        )

    if analyze:
        article = st.session_state.article.strip()
        if not article:
            st.warning("Please enter a news article or claim.")
        elif model is None:
            st.error(
                "The trained model could not be loaded. "
                "Run `python src/train.py` to create it."
            )
            if model_error:
                st.caption(model_error)
        else:
            with st.spinner("Analyzing text with the ML model..."):
                result = predict_news(article, model)
                record = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "input": article,
                    "prediction": str(result["prediction"]),
                    "scores": result["class_scores"],
                }
                st.session_state.history.insert(0, record)
                st.session_state.history = st.session_state.history[:50]
                st.session_state.last_result = record

    result = st.session_state.get("last_result")
    if result:
        st.subheader("Analysis results")
        prediction = result["prediction"]
        scores = result["scores"]

        if prediction.upper() == "FAKE":
            st.warning(f"### ML prediction: {prediction}")
        elif prediction.upper() == "REAL":
            st.success(f"### ML prediction: {prediction}")
        else:
            st.info(f"### ML prediction: {prediction}")

        st.markdown(
            """
            <div class="result-card">
                <div class="eyebrow">Interpretation</div>
                <div class="result-title">Model classification</div>
                <p class="muted">
                    This label reflects patterns learned from the training data.
                    It does not establish whether a claim is true or false.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Model class scores")
        score_columns = st.columns(max(1, len(scores)))
        for index, (label, score) in enumerate(scores.items()):
            with score_columns[index]:
                st.metric(label, f"{score * 100:.1f}%")
                st.progress(max(0.0, min(1.0, score)))
        st.caption("Scores are model outputs, not verified truth probabilities.")

        st.markdown("#### Evidence verification")
        st.info(
            "Source retrieval is not connected. No sources have been checked, "
            "so this app cannot provide an evidence-based verdict."
        )

elif page == "Analysis History":
    st.title("Analysis history")
    st.markdown(
        '<p class="muted">Predictions stored during this browser session.</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("No analyses yet. Start by analyzing a news article.")
    else:
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.pop("last_result", None)
            st.rerun()

        for item in st.session_state.history:
            with st.expander(f"{item['time']} · {item['prediction']}"):
                st.write(item["input"])
                st.write("Model scores:", item["scores"])

elif page == "About Project":
    st.title("About Verity AI")
    st.markdown(
        """
        <div class="result-card">
            <div class="eyebrow">Project overview</div>
            <div class="result-title">AI Fake News Detection &amp; Verification System</div>
            <p class="muted">
                An educational Python project that uses TF-IDF and Logistic
                Regression to classify news text. Evidence retrieval and
                RAG-based verification are planned, but are not implemented yet.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Technology stack")
    technologies = [
        ("Python", "Core programming language"),
        ("Scikit-learn", "TF-IDF and machine-learning classification"),
        ("Streamlit", "Interactive web application"),
        ("RAG", "Planned evidence retrieval and grounded explanations"),
    ]
    for name, description in technologies:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(description)

    st.subheader("Limitations")
    st.write(
        "The classifier can make mistakes. Its scores are not guaranteed to be "
        "calibrated probabilities, and its predictions are not proof of factual "
        "accuracy. Use reliable sources and human judgment to verify important claims."
    )
    st.info(
        "This application is an educational text-classification demo, not an "
        "authoritative fact-checking service."
    )
