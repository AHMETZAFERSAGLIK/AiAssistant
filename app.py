# app.py
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained model for intent classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '')
    result = classifier(text)
    intent = result[0]['label']

    # Basic intent handling
    if 'PRODUCT' in text.upper():
        reply = "Sure, what product would you like to know about?"
    elif 'APPOINTMENT' in text.upper():
        reply = "I can help you book an appointment. What date and time work for you?"
    else:
        reply = "Sorry, I didn't understand that."

    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
