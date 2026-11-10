
import json
import random
from pathlib import Path

from src.model import predict_intent


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "intents.json"


# ==========================================
# LOAD CHATBOT RESPONSES
# ==========================================

def load_responses():

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    responses = {}

    for intent in data["intents"]:

        tag = intent["tag"]
        intent_responses = intent["responses"]

        responses[tag] = intent_responses

    return responses


# ==========================================
# GENERATE CHATBOT RESPONSE
# ==========================================

def get_response(user_input, responses):

    # Predict the user's intent
    predicted_intent, confidence = predict_intent(user_input)

    # Convert NumPy values into standard Python types
    predicted_intent = str(predicted_intent)
    confidence = float(confidence)

    # Handle uncertain predictions
    CONFIDENCE_THRESHOLD = 0.35

    if confidence < CONFIDENCE_THRESHOLD:

        return (
            "I'm sorry, I didn't quite understand that. "
            "Could you please rephrase your question?"
        )

    # Find responses for the predicted intent
    possible_responses = responses.get(predicted_intent)

    if not possible_responses:

        return "I'm sorry, I don't have an answer for that."

    # Select a random response
    chatbot_response = random.choice(possible_responses)

    return chatbot_response


# ==========================================
# RUN CHATBOT
# ==========================================

def start_chatbot():

    responses = load_responses()

    print("=" * 50)
    print("UNIVERSITY ASSISTANT CHATBOT")
    print("=" * 50)

    print("Hello! I am your university assistant.")
    print("How can I help you today?")
    print("Type 'exit' to end the conversation.")

    print("-" * 50)

    while True:

        user_input = input("\nYou: ").strip()

        # Check for empty input
        if not user_input:

            print("Bot: Please enter a message.")
            continue

        # Exit chatbot
        if user_input.lower() in ["exit", "quit", "bye"]:

            print("Bot: Goodbye! Have a great day!")
            break

        # Generate response
        chatbot_response = get_response(
            user_input,
            responses
        )

        print("Bot:", chatbot_response)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    start_chatbot()
