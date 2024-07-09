import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pickle

# Load the dataset
data = pd.read_csv('email.csv', encoding='latin-1')
data = data[['Category', 'Message']]
data.columns = ['label', 'message']
data.dropna(subset=['label', 'message'], inplace=True)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
data.dropna(subset=['label'], inplace=True)

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

data['message'] = data['message'].apply(preprocess_text)

# Split the data
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes pipeline
nb_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Support Vector Machine pipeline
svm_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='linear')),
])

# Train models
nb_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# Save models
with open('nb_model.pkl', 'wb') as nb_file:
    pickle.dump(nb_pipeline, nb_file)

with open('svm_model.pkl', 'wb') as svm_file:
    pickle.dump(svm_pipeline, svm_file)
