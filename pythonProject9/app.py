from flask import Flask, request, render_template
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

# Load pre-trained models
nb_pipeline = pickle.load(open('nb_model.pkl', 'rb'))
svm_pipeline = pickle.load(open('svm_model.pkl', 'rb'))

# Text preprocessing function
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']
        processed_message = preprocess_text(message)
        nb_prediction = nb_pipeline.predict([processed_message])[0]
        svm_prediction = svm_pipeline.predict([processed_message])[0]

        return render_template('index.html', message=message,
                               nb_result='Spam' if nb_prediction == 1 else 'Ham',
                               svm_result='Spam' if svm_prediction == 1 else 'Ham')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
