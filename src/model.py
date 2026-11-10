
import joblib
from pathlib import Path

from src.preprocessing import clean_text


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "intent_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python -m src.train first."
        )

    model = joblib.load(MODEL_PATH)

    return model


# ==========================================
# LOAD TF-IDF VECTORIZER
# ==========================================

def load_vectorizer():

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "TF-IDF vectorizer not found. Run python -m src.train first."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)

    return vectorizer


# ==========================================
# PREDICT USER INTENT
# ==========================================

def predict_intent(user_input):

    # Step 1: Clean user input
    cleaned_text = clean_text(user_input)

    # Step 2: Load trained model and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()

    # Step 3: Convert text into TF-IDF features
    text_vector = vectorizer.transform([cleaned_text])

    # Step 4: Predict intent
    predicted_intent = model.predict(text_vector)[0]

    # Step 5: Get prediction confidence
    probabilities = model.predict_proba(text_vector)[0]

    confidence = max(probabilities)

    return predicted_intent, confidence
