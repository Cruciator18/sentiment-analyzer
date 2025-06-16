import streamlit as st
import requests
from textblob import TextBlob
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive", polarity
    elif polarity < 0:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Fetch articles from NewsAPI
def fetch_news(query, max_articles=10):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': query,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'pageSize': max_articles
    }

    response = requests.get(url, params=params)
    results = []

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        for article in articles:
            title = article['title']
            sentiment, polarity = analyze_sentiment(title)
            results.append({
                "title": title,
                "source": article['source']['name'],
                "url": article['url'],
                "sentiment": sentiment,
                "polarity": round(polarity, 2)
            })
    else:
        st.error(f"API Error {response.status_code}: {response.text}")
    return results

# ------------------------------------------
# Streamlit UI
# ------------------------------------------

st.set_page_config(page_title="📰 News Sentiment Analyzer", layout="wide")
st.title("🧠 Real-time News Sentiment Analyzer")
st.caption("Analyze news headlines and detect overall sentiment 💬")

query = st.text_input("🔍 Enter a keyword or topic", placeholder="e.g. AI, climate change, cricket...")

if query:
    with st.spinner("Fetching news and analyzing sentiment..."):
        articles = fetch_news(query)

    if articles:
        # Convert to DataFrame
        df = pd.DataFrame(articles)

        # Sentiment counts
        sentiment_counts = df['sentiment'].value_counts()

        # Display summary
        with st.container():
            st.subheader("📊 Sentiment Summary")
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'], ax=ax)
                plt.title("Sentiment Distribution")
                plt.xticks(rotation=0)
                st.pyplot(fig)

            with col2:
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    count = sentiment_counts.get(sentiment, 0)
                    emoji = "🟢" if sentiment == "Positive" else "⚪" if sentiment == "Neutral" else "🔴"
                    st.metric(label=f"{emoji} {sentiment}", value=count)

        # Article display
        st.subheader("🗞️ News Articles")
        for _, row in df.iterrows():
            sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}.get(row['sentiment'], "❓")
            with st.expander(f"{sentiment_emoji} {row['title']}"):
                st.write(f"**Sentiment:** {row['sentiment']} (Polarity: {row['polarity']})")
                st.write(f"**Source:** {row['source']}")
                st.markdown(f"[🔗 Read Full Article]({row['url']})")
                st.markdown("---")
    else:
        st.warning("No articles found. Try a different keyword.")
