from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
try:
    with open('model/imdb_sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded'}), 500
    
    review = request.json.get('review', '').strip()
    if not review:
        return jsonify({'error': 'Empty review'}), 400
    
    # Predict
    vec = vectorizer.transform([review])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    sentiment = 'Positive 😊' if pred == 1 else 'Negative 😞'
    confidence = round(prob[pred] * 100, 1)
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
