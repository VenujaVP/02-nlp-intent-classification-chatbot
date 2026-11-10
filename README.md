# NLP Intent Classification Chatbot

A machine learning-based chatbot that understands user messages and identifies their intents using Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression.

The chatbot is designed as a university assistant that can recognize common user intentions such as greetings, course inquiries, fee inquiries, opening hours, thanks, and goodbyes.

## Features

- **Text preprocessing:** Converts text to lowercase and removes punctuation and unnecessary spaces.
- **Intent classification:** Uses a Logistic Regression machine learning model to classify user messages.
- **TF-IDF vectorization:** Converts text into numerical features for machine learning.
- **Multiple intents:** Supports six different intent categories.
- **Response generation:** Selects a response from the matching intent in the dataset.
- **Confidence threshold:** Handles predictions with low confidence by asking the user to rephrase.
- **Model persistence:** Saves and loads the trained model and TF-IDF vectorizer using Joblib.

## Technologies Used

- Python
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- JSON

## Project Structure

```text
02-nlp-intent-classification-chatbot/
│
├── src/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── train.py
│   ├── preprocessing.py
│   └── model.py
│
├── data/
│   └── intents.json
│
├── models/
│   ├── intent_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_tfidf.py
│   └── test_dataset.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The chatbot uses a JSON dataset containing example sentences and responses for each intent.

Currently, the dataset contains **120 example sentences across 6 intent categories**.

| Intent | Number of examples |
|---|---:|
| greeting | 20 |
| courses | 20 |
| fees | 20 |
| opening_hours | 20 |
| thanks | 20 |
| goodbye | 20 |
| **Total** | **120** |

Each intent contains:
- A unique intent tag.
- Example user sentences (patterns).
- Possible chatbot responses.

## Machine Learning Workflow

The project follows this workflow:

1. Load the intent dataset from the JSON file.
2. Clean and preprocess the text.
3. Split the dataset into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression classifier.
6. Predict intents using the trained model.
7. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Save the trained model and TF-IDF vectorizer.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

### 2. Navigate to the project directory

```bash
cd 02-nlp-intent-classification-chatbot
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Train the machine learning model

Run:

```bash
python -m src.train
```

This trains the Logistic Regression classifier and saves the model and TF-IDF vectorizer inside the `models/` directory.

### Step 2: Start the chatbot

Run:

```bash
python -m src.chatbot
```

You can now enter messages and interact with the chatbot through the terminal.

To end the conversation, type:

```text
exit
```

## Model Evaluation

The current model was evaluated using an 80:20 training/testing split.

| Evaluation detail | Result |
|---|---:|
| Total examples | 120 |
| Training examples | 96 |
| Testing examples | 24 |
| Model | Logistic Regression |
| Vectorization | TF-IDF |
| Accuracy | **83.33%** |

The model correctly classified 20 out of 24 test examples in the current evaluation.

**Note:** This accuracy is based on one train/test split with a relatively small dataset. It should not be considered a final measure of real-world performance. Further testing and dataset improvements are needed.

## Running Tests

The project includes tests for preprocessing, TF-IDF vectorization, and dataset validation.

Run the dataset validation test:

```bash
python -m tests.test_dataset
```

Run the preprocessing test:

```bash
python -m tests.test_preprocessing
```

Run the TF-IDF test:

```bash
python -m tests.test_tfidf
```

## Future Improvements

- Expand the dataset with more diverse training examples.
- Improve classification performance, especially for similar intents.
- Evaluate the model using cross-validation.
- Improve confidence threshold handling.
- Add more university-related intents.
- Develop a graphical user interface or web-based chatbot.
- Explore more advanced NLP and machine learning approaches.

## Author

**[Your Name]**

GitHub: [Your GitHub Profile](<YOUR-GITHUB-PROFILE-URL>)

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.