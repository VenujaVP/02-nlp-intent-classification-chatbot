
from src.preprocessing import clean_text


message = "  HELLO!!!   How Are You?  "

result = clean_text(message)

print("Original:", message)
print("Cleaned:", result)
