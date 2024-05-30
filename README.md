# Text-Emotion-Analysis
Text emotion analysis is to classify texts into five emotion categories: joy, sadness, anger, fear, neutral. A project to go through different text classification techniques. This includes dataset preparation, traditional machine learning with scikit-learn, LSTM neural networks and transfer learning using BERT (tensorflow keras).
Data preprocessing: noise and punctuation removal, tokenization, stemming
Text Representation: TF-IDF
Classifiers: Naive Bayes, Random Forrest, Logistic Regrassion, SVM

The project involves building an emotion classification system for short messages using Scikit-learn. The process begins with importing and preparing the dataset, which is divided into training and testing sets. The dataset contains messages labeled with five emotions: joy, sadness, anger, fear, and neutral. The preprocessing step includes cleaning the text data by removing HTML tags, URLs, hashtags, mentions, punctuation, and digits. Tokenization is then performed to split the text into individual words, followed by stemming to reduce words to their root forms. This ensures the text data is in a consistent format suitable for analysis.

Next, the cleaned and tokenized text data is vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) method, which converts the text into numerical feature vectors. The TF-IDF representation considers both the frequency of words in a document and the importance of words across the entire corpus, thus capturing the significance of words in the context of the dataset. Various classifiers are then trained on the vectorized text data, including Multinomial Naive Bayes, Random Forest, Logistic Regression, and Support Vector Machine (SVM). These models are evaluated based on their accuracy and F1 scores to determine their effectiveness in classifying emotions.

The project culminates in the evaluation of the classifiers using confusion matrices, which provide insights into the performance of each model by showing the number of correct and incorrect predictions for each emotion class. The SVM classifier, which demonstrated the highest accuracy and F1 score, is selected as the best-performing model. Finally, this SVM model is saved using Python's pickle module for future use. By loading the saved model, it can be tested on new, unseen messages to predict their emotional content, showcasing the practical application of the emotion classification system.






