import streamlit as st
import requests


import pandas as pd

from sentiment import analyze_sentiment  

# Load environment variables

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]


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
            result = analyze_sentiment(title)
            sentiment = {"POSITIVE": "Positive", "NEGATIVE": "Negative"}.get(result["label"].upper(), "Neutral")
            score = result["score"]
            results.append({
                "title": title,
                "source": article['source']['name'],
                "url": article['url'],
                "sentiment": sentiment,
                "polarity": score  
            })
    else:
        st.error(f"API Error {response.status_code}: {response.text}")
    return results


# Streamlit UI


st.set_page_config(page_title="📰 News Sentiment Analyzer", layout="wide")
st.title("🧠 Real-time News Sentiment Analyzer")
st.caption("Analyze news headlines using a deep learning model 💬")

query = st.text_input("🔍 Enter a keyword or topic", placeholder="e.g. AI, climate change, cricket...")

if query:
    with st.spinner("Fetching news and analyzing sentiment..."):
        articles = fetch_news(query)

    if articles:
        df = pd.DataFrame(articles)
        sentiment_counts = df['sentiment'].value_counts()

        # Sentiment summary
        with st.container():
            st.subheader("📊 Sentiment Summary")
            col1, col2 = st.columns(2)

            with col1:
                ordered_counts= pd.DataFrame({
                    "Sentiment" : ["Positive", "Negative","Neutral"],
                    "Count" : [sentiment_counts.get("Positive", 0),
                               sentiment_counts.get("Negative,0"),
                               sentiment_counts.get("Neutral,0"),
                               ]
                })
                
                ordered_counts.set_index("Sentiment", inplace=True)
                st.bar_chart(ordered_counts)

            with col2:
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    count = sentiment_counts.get(sentiment, 0)
                    emoji = "🟢" if sentiment == "Positive" else "⚪" if sentiment == "Neutral" else "🔴"
                    st.metric(label=f"{emoji} {sentiment}", value=count)

        # Show articles
        st.subheader("🗞️ News Articles")
        for _, row in df.iterrows():
            sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}.get(row['sentiment'], "❓")
            with st.expander(f"{sentiment_emoji} {row['title']}"):
                st.write(f"**Sentiment:** {row['sentiment']} (Confidence: {row['polarity']}%)")
                st.write(f"**Source:** {row['source']}")
                st.markdown(f"[🔗 Read Full Article]({row['url']})")
                st.markdown("---")
    else:
        st.warning("No articles found. Try a different keyword.")
