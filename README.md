# **IMDB Sentiment Analysis – Flask App**

A simple Flask web application that predicts whether a movie review is **Positive** or **Negative** using a trained **Multinomial Naive Bayes** model.

---

## **Features**

* Train a sentiment analysis model using the IMDB dataset
* Uses `CountVectorizer` for text processing
* Flask API for predictions
* Saves model and vectorizer as `.pkl` files

---

## **Project Structure**

```
Movie_Review_Checker/
├── app.py
├── model_training.py
├── model/
│   ├── imdb_sentiment_model.pkl
│   └── vectorizer.pkl
├── data/
│   └── IMDB Dataset.csv
├── templates/
│   └── index.html
└── README.md
```

---

## **How to Train the Model**

```
python train.py
```

This generates:

* `imdb_sentiment_model.pkl`
* `vectorizer.pkl`
* `confusion_matrix.png`

---

## **How to Run the Flask App**

```
python app.py
```

App runs on: **[http://localhost:5000](http://localhost:5000)**

---

## **API Endpoint**

### POST `/predict`

**Body:**

```json
{
  "review": "The movie was great!"
}
```

**Response:**

```json
{
  "sentiment": "Positive 😊",
  "confidence": 92.5
}
```

---

## **Requirements**

```
Flask
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## **License**

Free to use and modify.
