# PRODIGY_INFOTECH_DATA_SCIENCE_INTERNSHIP_TASK_04

# Task-04: Sentiment Analysis of Social Media Data

## Task Description
Analyze and visualize sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands.

## Dataset
You can access the Twitter Entity Sentiment Analysis dataset from Kaggle here:
[Twitter Entity Sentiment Analysis Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

## Instructions
1. Download the dataset from Kaggle.
2. Load the dataset using `pandas`.
3. Perform **Data Preprocessing**:
   - Remove missing values.
   - Clean the text data (remove special characters, stopwords, and perform tokenization).
   - Convert text into numerical features using TF-IDF or word embeddings.
4. **Sentiment Analysis**:
   - Use NLP techniques to classify sentiment into positive, negative, or neutral.
   - Train a machine learning model such as Logistic Regression, Naive Bayes, or an LSTM-based deep learning model.
5. **Data Visualization**:
   - Use bar plots, word clouds, and sentiment distribution graphs to analyze public opinion.
   - Plot trends in sentiment over time.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `nltk`
  - `sklearn`
  - `matplotlib`
  - `seaborn`
  - `wordcloud`
  - `tensorflow` (if using deep learning)

## Example Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("twitter_sentiment.csv")  # Use the correct file name

# Drop missing values
df.dropna(inplace=True)

# Text Preprocessing
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word.isalpha() and word not in stop_words]))

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']  # Assuming sentiment column has labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Visualize sentiment distribution
plt.figure(figsize=(8,5))
sns.countplot(x=y)
plt.title("Sentiment Distribution")
plt.show()

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_text']))
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis
