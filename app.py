from flask import Flask, render_template, request
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

app = Flask(__name__)

# Load the trained model
model = load_model('model/fake_news_model.h5')
voc_size = 5000
sent_length = 20

# Preprocessing function
def preprocess_news(news_title):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', news_title)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

# Prediction function
def predict_fake_news(news_title):
    processed_title = preprocess_news(news_title)
    onehot_repr = one_hot(processed_title, voc_size)
    embedded_doc = pad_sequences([onehot_repr], padding='pre', maxlen=sent_length)
    prediction = model.predict(embedded_doc)
    return prediction[0][0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        news_article = request.form['news']
        probability = predict_fake_news(news_article)
        result = 'Real' if probability > 0.5 else 'Fake'
        return render_template('index.html', result=result, probability=probability)
    return render_template('index.html', result='', probability='')

if __name__ == '__main__':
    app.run(debug=True)
