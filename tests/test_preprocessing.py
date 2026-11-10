
from src.preprocessing import clean_text, tokenize_text


message = "  HELLO!!!   How Are You?  "

# Step 1: Clean the text
cleaned_text = clean_text(message)

# Step 2: Tokenize the cleaned text
tokens = tokenize_text(cleaned_text)

print("Original:", message)
print("Cleaned:", cleaned_text)
print("Tokens:", tokens)
