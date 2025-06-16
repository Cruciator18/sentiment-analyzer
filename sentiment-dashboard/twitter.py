import requests
from textblob import TextBlob


NEWS_API_KEY = "e85f384a9a574c72a3b5598b24ac3c01"


def create_headers(token):
    return {"Authorization": f"Bearer {token}"}

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

def search_recent_tweets(query, max_results=10):
    url = "https://api.twitter.com/2/tweets/search/recent"
    query_params = {
        'query': query,
        'max_results': max_results,
        'tweet.fields': 'created_at,author_id'
    }

    headers = create_headers(BEARER_TOKEN)
    response = requests.get(url, headers=headers, params=query_params)

    if response.status_code == 200:
        tweets = response.json().get('data', [])
        for tweet in tweets:
            sentiment, score = analyze_sentiment(tweet['text'])
            print(f"{tweet['created_at']} - @{tweet['author_id']}:")
            print(tweet['text'])
            print(f"Sentiment: {sentiment} (Score: {score:.2f})\n")
    else:
        print("Failed to fetch tweets:", response.status_code, response.text)

if __name__ == "__main__":
    search_query = input("Enter search query: ")
    search_recent_tweets(search_query)
e85f384a9a574c72a3b5598b24ac3c01