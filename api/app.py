import json
from flask import Flask, request, render_template
from transformers import pipeline
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '../templates'))


# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# File path for storing reviews
reviews_file = 'reviews.json'

# Load existing reviews from the JSON file
def load_reviews():
    try:
        with open(reviews_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'positive': [], 'negative': [], 'neutral': []}

# Save reviews to the JSON file
def save_reviews(reviews):
    with open(reviews_file, 'w') as f:
        json.dump(reviews, f, indent=4)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    reviews = load_reviews()  # Load existing reviews from the JSON file

    if request.method == 'POST':
        review = request.form['review']
        result = sentiment_pipeline(review)[0]
        sentiment_label = result['label'].lower()

        # Categorize the review into positive, negative, or neutral
        if sentiment_label == 'positive':
            reviews['positive'].append({'review': review, 'confidence': f"{result['score'] * 100:.2f}%"})
        elif sentiment_label == 'negative':
            reviews['negative'].append({'review': review, 'confidence': f"{result['score'] * 100:.2f}%"})
        else:
            reviews['neutral'].append({'review': review, 'confidence': f"{result['score'] * 100:.2f}%"})

        # Save the updated reviews to the JSON file
        save_reviews(reviews)

        # Return the sentiment analysis result
        sentiment = {
            'label': result['label'],
            'score': f"{result['score'] * 100:.2f}%"  # Convert to percentage with 2 decimal places
        }

    return render_template("index.html", sentiment=sentiment, reviews=reviews)

if __name__ == "__main__":
    app.run(debug=True)
