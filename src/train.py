
import json
from pathlib import Path

import joblib

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocessing import clean_text


# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset and model storage locations
DATA_PATH = BASE_DIR / "data" / "intents.json"
MODEL_DIR = BASE_DIR / "models"


def load_dataset():
    # Open the JSON dataset
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    sentences = []
    labels = []

    # Extract sentences and their intent labels
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            sentences.append(clean_text(pattern))
            labels.append(intent["tag"])

    return sentences, labels


def train_model():
    # Load the training dataset
    sentences, labels = load_dataset()

    print("Total examples:", len(sentences))

    # Divide data into training and testing groups
    X_train, X_test, y_train, y_test = train_test_split(
        sentences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print("Training examples:", len(X_train))
    print("Testing examples:", len(X_test))

    # Convert text into TF-IDF numerical features
    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create the Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train_tfidf, y_train)

    print("\nModel training completed!")


    # Test the model
    predictions = model.predict(X_test_tfidf)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation")
    print("-------------------------")

    print("Accuracy:", round(accuracy * 100, 2), "%")

    # Calculate precision, recall and F1-score
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=model.classes_,
            zero_division=0
        )
    )

    # Calculate confusion matrix
    print("Confusion Matrix:")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    print("Intent order:", model.classes_)
    print(matrix)


    # Save the trained model and vectorizer
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "intent_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

    print("\nModel and vectorizer saved successfully!")


if __name__ == "__main__":
    train_model()
