import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
# from gui import input
nltk.download('stopwords')
import pickle as pkl

portStem = PorterStemmer()

def stemming(content):
    stemed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemed_content = stemed_content.lower()
    stemed_content = stemed_content.split()
    stemed_content = [portStem.stem(word) for word in stemed_content if not word in stopwords.words('english')]
    stemed_content = ' '.join(stemed_content)
    return stemed_content

# Load dataset
try:
    dataset = pd.read_csv('F:\\ML Projects\\Fake news prediction\\fake_or_real_news.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")

# print(dataset.head())
# Check dataset
dataset['title'] = dataset['title'].apply(stemming)

X = dataset['title'].values
Y = dataset['label'].values

vectorizer = TfidfVectorizer() #term frequency and inverse document frequency
X = vectorizer.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, stratify= Y,random_state=2)
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
tarin_data_accuracy = accuracy_score(X_train_prediction, Y_train)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Training data accuracy :",tarin_data_accuracy,"Test data accuracy :", test_data_accuracy)
X_new = X_test[0]

pkl.dump( model,open("Model.pkl", "wb"))
with open("Vectorizer.pkl", "wb") as file:
    pkl.dump(vectorizer, file)

