# 📰 Fake News Prediction using Machine Learning

This project implements a Fake News Detection system using two powerful models: **Logistic Regression with TF-IDF vectorization** and a **Deep Learning-based LSTM model**. It classifies news articles as **real** or **fake** based on their text content.

---

## 📌 Features

- Data preprocessing with NLTK: tokenization, stopword removal, and lemmatization
- Dual model approach:
  - Logistic Regression (TF-IDF)
  - LSTM (Neural Network with Embedding Layer)
- Model evaluation using accuracy, precision, recall, F1-score
- Confusion matrix visualizations for both models
- Sample prediction demo

---

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy**
- **NLTK** – natural language preprocessing
- **Scikit-learn** – ML model and evaluation metrics
- **TensorFlow / Keras** – LSTM neural network
- **Matplotlib & Seaborn** – visualization

---

## 📂 Dataset Source
- Download the dataset from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- Place `True.csv` and `Fake.csv` in your project directory

The project uses two separate CSV files:

- `True.csv` – Real news articles
- `Fake.csv` – Fake or misleading articles

These files are merged and labeled for supervised learning.

---

## 🧪 How the Project Works

### 1. **Data Preprocessing**
- Combine title and content
- Clean text (remove URLs, punctuation, lowercasing)
- Tokenization, stopword removal, and lemmatization

### 2. **Model Training**

#### 🔹 Logistic Regression
- Uses **TF-IDF Vectorizer** to convert text into features
- Trained using `sklearn.linear_model.LogisticRegression`

#### 🔹 LSTM Model
- Uses **Tokenizer** and **pad_sequences** to prepare sequences
- Embedding → LSTM → Dense layers
- Compiled with binary crossentropy loss and Adam optimizer

### 3. **Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix for each model

### 4. **Demo Prediction**
- User can input sample news text
- Prediction shown from both models (Real or Fake)



