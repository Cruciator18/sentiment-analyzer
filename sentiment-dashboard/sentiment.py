from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch

@st.cache_resource  # ✅ Cache model loading across reruns
def load_sentiment_pipeline():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = 0 if torch.cuda.is_available() else -1  # ✅ Use CPU (-1) on Streamlit Cloud
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

sentiment_pipeline = load_sentiment_pipeline()

def analyze_sentiment(text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    result = sentiment_pipeline(text[:512])[0]
    return {
        "label": result["label"],
        "score": round(result["score"] * 100, 2)
    }
