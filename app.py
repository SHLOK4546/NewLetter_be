from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from nltk.tokenize import sent_tokenize
from flask_cors import CORS
from safetensors.torch import load_file
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure NLTK resources are available
try:
    sent_tokenize("Test sentence.")
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load FinBERT model and tokenizer with safetensors
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", use_safetensors=True)

# Sentiment labels mapping
labels = {0: "positive", 1: "negative", 2: "neutral"}

def analyze_sentiment(text):
    """Analyze sentiment of input text using FinBERT"""
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    if not sentences:
        return {"error": "No valid text provided"}
    
    # Process each sentence
    results = []
    for sentence in sentences:
        # Tokenize and prepare input
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
        results.append({
            "sentence": sentence,
            "sentiment": labels[prediction],
            "confidence": float(confidence)
        })
    
    # Calculate overall sentiment
    sentiments = [r["sentiment"] for r in results]
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    
    # Determine overall sentiment
    positive_count = sentiments.count("positive")
    negative_count = sentiments.count("negative")
    neutral_count = sentiments.count("neutral")
    
    if positive_count > negative_count and positive_count > neutral_count:
        overall_sentiment = "positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"
        
    return {
        "overall_sentiment": overall_sentiment,
        "average_confidence": float(avg_confidence),
        "detailed_results": results
    }

@app.route('/api/sentiment', methods=['POST'])
def sentiment_analysis():
    """API endpoint to analyze sentiment of news text"""
    try:
        data = request.get_json()
        news_text = data.get('news_text')
        
        if not news_text:
            return jsonify({"error": "No news text provided"}), 400
            
        result = analyze_sentiment(news_text)
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)