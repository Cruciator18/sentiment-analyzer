from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create sentiment analysis pipeline with device
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

def analyze_sentiment(text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    result = sentiment_pipeline(text[:512])[0]
    return {
        "label": result["label"],
        "score": round(result["score"] * 100, 2)
    }
