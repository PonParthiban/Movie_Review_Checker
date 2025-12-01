import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import pickle

# Load data
data = pd.read_csv("data/IMDB Dataset.csv")
df = pd.DataFrame(data)

# Clean duplicates
dup_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"Removed duplicates: {dup_before}")
print(f"Null values:\n{df.isnull().sum()}")

# Encode sentiment
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X = df['review']
y = df['sentiment']

# Split BEFORE vectorization (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Vectorize on training data only
vec = CountVectorizer()
X_train_vectorized = vec.fit_transform(X_train)
X_test_vectorized = vec.transform(X_test)  # transform only, don't fit

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vectorized)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Save confusion matrix plot instead of showing
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - IMDB Sentiment Classification')
plt.savefig("confusion_matrix.png")

# Save model 
with open("imdb_sentiment_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vec, vec_file)
    


