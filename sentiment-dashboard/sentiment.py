from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_sentiment_pipeline():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

def analyze_sentiment(pipeline, text):
    if not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}
    result = pipeline(text[:512])[0]
    return {
        "label": result["label"],
        "score": round(result["score"] * 100, 2)
    }
