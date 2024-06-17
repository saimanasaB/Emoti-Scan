import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle

# Load datasets
df_train = pd.read_csv('data/data_train.csv')
df_test = pd.read_csv('data/data_test.csv')

X_train = df_train.Text
X_test = df_test.Text
y_train = df_train.Emotion
y_test = df_test.Emotion

class_names = ['joy', 'sadness', 'anger', 'neutral', 'fear']
data = pd.concat([df_train, df_test])

# Preprocess and tokenize function
def preprocess_and_tokenize(data):    
    data = re.sub("(<.*?>)", "", data)
    data = re.sub(r'http\S+', '', data)
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    data = re.sub("(\\W|\\d)", " ", data)
    data = data.strip()
    data = word_tokenize(data)
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]
    return stem_data

# Plot confusion matrix function
def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)

# TFIDF Vectorizer
vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize, sublinear_tf=True, norm='l2', ngram_range=(1, 2))
vect.fit_transform(data.Text)
X_train_vect = vect.transform(X_train)
X_test_vect = vect.transform(X_test)

# Train models and display results
def train_and_evaluate_model(model, model_name):
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred, average='micro') * 100
    cm = confusion_matrix(y_test, y_pred)
    
    st.write(f"{model_name} Accuracy: {accuracy:.2f}%")
    st.write(f"{model_name} F1 Score: {f1:.2f}")
    plot_confusion_matrix(cm, classes=class_names, normalize=True, title=f'{model_name} Normalized Confusion Matrix')

# Streamlit app
st.title('Emotion Classification in Texts using Scikit-learn')
st.write("## Load and Evaluate Models")

if st.checkbox("Train and Evaluate Models"):
    st.write("### Naive Bayes")
    nb = MultinomialNB()
    train_and_evaluate_model(nb, "Naive Bayes")
    
    st.write("### Random Forest")
    rf = RandomForestClassifier(n_estimators=50)
    train_and_evaluate_model(rf, "Random Forest")
    
    st.write("### Logistic Regression")
    log = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200)
    train_and_evaluate_model(log, "Logistic Regression")
    
    st.write("### Linear SVM")
    svc = LinearSVC(tol=1e-05)
    train_and_evaluate_model(svc, "Linear SVM")

    # Saving the tf-idf + SVM Model
    svm_model = Pipeline([
        ('tfidf', vect),
        ('clf', svc),
    ])
    filename = 'models/tfidf_svm.sav'
    pickle.dump(svm_model, open(filename, 'wb'))
    model = pickle.load(open(filename, 'rb'))

st.write("## Predict Emotion in New Text")
message = st.text_input("Enter a message to predict its emotion:")

if message:
    model = pickle.load(open('models/tfidf_svm.sav', 'rb'))
    prediction = model.predict([message])
    st.write(f"The predicted emotion is: {prediction[0]}")
