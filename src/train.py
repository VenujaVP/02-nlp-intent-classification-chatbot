
import json
from pathlib import Path

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.preprocessing import clean_text


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "intents.json"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "intent_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    sentences = []
    labels = []

    for intent in data["intents"]:

        for pattern in intent["patterns"]:

            sentences.append(clean_text(pattern))
            labels.append(intent["tag"])

    return sentences, labels


# ==========================================
# TRAIN AND EVALUATE MODEL
# ==========================================

def train_model():

    # Step 1: Load dataset
    sentences, labels = load_dataset()

    print("=" * 50)
    print("INTENT CLASSIFICATION CHATBOT")
    print("=" * 50)

    print("\nDataset Information")
    print("-" * 50)

    print("Total examples:", len(sentences))
    print("Total intents:", len(set(labels)))

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        sentences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    print("Training examples:", len(X_train))
    print("Testing examples:", len(X_test))

    # Step 3: TF-IDF vectorization
    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTF-IDF Information")
    print("-" * 50)

    print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

    # Step 4: Create Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Step 5: Train model
    model.fit(X_train_tfidf, y_train)

    print("\nModel training completed!")

    # Step 6: Make predictions
    predictions = model.predict(X_test_tfidf)

    # ==========================================
    # PREDICTION ANALYSIS
    # ==========================================

    print("\nPrediction Analysis")
    print("=" * 50)

    correct_predictions = 0
    incorrect_predictions = 0

    for sentence, actual, predicted in zip(
        X_test,
        y_test,
        predictions
    ):

        print("\nSentence:", sentence)
        print("Actual Intent:", actual)
        print("Predicted Intent:", predicted)

        if actual == predicted:

            print("Result: CORRECT")
            correct_predictions += 1

        else:

            print("Result: INCORRECT")
            incorrect_predictions += 1

    print("\nPrediction Summary")
    print("-" * 50)

    print("Correct predictions:", correct_predictions)
    print("Incorrect predictions:", incorrect_predictions)

    # ==========================================
    # MODEL EVALUATION
    # ==========================================

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation")
    print("=" * 50)

    print("Accuracy:", round(accuracy * 100, 2), "%")

    # Classification report
    print("\nClassification Report")
    print("-" * 50)

    print(
        classification_report(
            y_test,
            predictions,
            labels=model.classes_,
            zero_division=0
        )
    )

    # Confusion matrix
    print("Confusion Matrix")
    print("-" * 50)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    print("Intent order:", model.classes_)
    print(matrix)

    # ==========================================
    # SAVE MODEL AND VECTORIZER
    # ==========================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel Saving")
    print("-" * 50)

    print("Model saved:", MODEL_PATH)
    print("Vectorizer saved:", VECTORIZER_PATH)

    print("\nTraining and evaluation completed successfully!")
    print("=" * 50)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    train_model()
