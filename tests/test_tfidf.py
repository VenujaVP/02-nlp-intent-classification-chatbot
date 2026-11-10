
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_text


# Example training sentences
sentences = [
    "What courses do you offer?",
    "Which courses are available?",
    "How much are the course fees?"
]


# Step 1: Clean the sentences
cleaned_sentences = [
    clean_text(sentence) for sentence in sentences
]


# Step 2: Create the TF-IDF tool
vectorizer = TfidfVectorizer()


# Step 3: Learn words and convert sentences into numbers
tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)


# Step 4: Display the results
print("Cleaned Sentences:")
print(cleaned_sentences)

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Numerical Values:")
print(tfidf_matrix.toarray().round(2))
