from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

app = Flask(__name__)

# Function to train the model
def train_model(df):
    # Get labels
    labels = df.label
    
    # Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(df['text'])

    # Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, labels)

    return tfidf_vectorizer, pac

# Read the data
df = pd.read_csv('E:\\fake news\\news.csv')

# Train the model
tfidf_vectorizer, pac = train_model(df)

# Function to predict using the trained model
def predict_fake_news(news_text):
    # Vectorize the input text
    tfidf_news = tfidf_vectorizer.transform([news_text])
    # Predict
    prediction = pac.predict(tfidf_news)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get news text from the form
    news_text = request.form['news_text']
    if not news_text.strip():
        return "Please enter some text."

    # Predict using the trained model
    prediction = predict_fake_news(news_text)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
