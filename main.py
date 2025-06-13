import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load and preprocess the dataset from two separate CSV files
def load_data(true_file_path, fake_file_path):
    """Load and merge true and fake news datasets."""
    # Load true news and assign label 0
    df_true = pd.read_csv(true_file_path)
    df_true['label'] = 0
    # Combine title and text into a single text column
    df_true['text'] = df_true['title'].astype(str) + ' ' + df_true['text'].astype(str)

    # Load fake news and assign label 1
    df_fake = pd.read_csv(fake_file_path)
    df_fake['label'] = 1
    df_fake['text'] = df_fake['title'].astype(str) + ' ' + df_fake['text'].astype(str)

    # Merge datasets
    df = pd.concat([df_true[['text', 'label']], df_fake[['text', 'label']]], ignore_index=True)

    return df['text'], df['label']

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters, URLs, and punctuation
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# 2. Prepare data for Logistic Regression (TF-IDF)
def prepare_tfidf_data(texts, labels):
    """Vectorize text data using TF-IDF."""
    # Apply preprocessing
    texts_processed = [preprocess_text(text) for text in texts]
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts_processed, labels, test_size=0.2, random_state=42
    )
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

# 3. Prepare data for LSTM (Tokenization and Padding)
def prepare_lstm_data(texts, labels, max_words=5000, max_len=100):
    """Tokenize and pad text data for LSTM."""
    # Apply preprocessing
    texts_processed = [preprocess_text(text) for text in texts]
    # Tokenize
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts_processed)
    sequences = tokenizer.texts_to_sequences(texts_processed)
    # Pad sequences
    X_padded = pad_sequences(sequences, maxlen=max_len)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, labels, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, tokenizer

# 4. Train and evaluate Logistic Regression
def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return y_pred, model

# 5. Train and evaluate LSTM
def train_lstm(X_train, X_test, y_train, y_test, max_words=5000, max_len=100):
    """Train and evaluate LSTM model."""
    model = Sequential([
        Embedding(max_words, 100, input_length=max_len),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\nLSTM Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return y_pred, model

# 6. Visualize confusion matrix
def plot_confusion_matrix(y_test, y_pred, title):
    """Plot confusion matrix using Seaborn."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load dataset (replace with your file paths)
    true_file_path = '/content/drive/MyDrive/Fake news/True.csv'  # Update with your true news file path
    fake_file_path = '/content/drive/MyDrive/Fake news/Fake.csv'  # Update with your fake news file path
    texts, labels = load_data(true_file_path, fake_file_path)

    # Prepare data for Logistic Regression
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = prepare_tfidf_data(texts, labels)

    # Train and evaluate Logistic Regression
    y_pred_lr, lr_model = train_logistic_regression(X_train_tfidf, X_test_tfidf, y_train, y_test)
    plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")

    # Prepare data for LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, tokenizer = prepare_lstm_data(texts, labels)

    # Train and evaluate LSTM
    y_pred_lstm, lstm_model = train_lstm(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)
    plot_confusion_matrix(y_test_lstm, y_pred_lstm, "LSTM Confusion Matrix")

    # Example: Predict a new article
    sample_text = ["This article claims aliens landed in New York, but no evidence supports it."]
    sample_processed = [preprocess_text(text) for text in sample_text]
    # For Logistic Regression
    sample_tfidf = vectorizer.transform(sample_processed)
    lr_prediction = lr_model.predict(sample_tfidf)
    print("\nSample Prediction (Logistic Regression):", "Fake" if lr_prediction[0] == 1 else "Real")
    # For LSTM
    sample_seq = tokenizer.texts_to_sequences(sample_processed)
    sample_padded = pad_sequences(sample_seq, maxlen=100)
    lstm_prediction = (lstm_model.predict(sample_padded) > 0.5).astype(int)
    print("Sample Prediction (LSTM):", "Fake" if lstm_prediction[0] == 1 else "Real")
