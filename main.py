import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

# Download NLTK resources if not already present

nltk.download('punkt')

nltk.download('stopwords')

# Function for Tokenization and Preprocessing

def tokenize_and_preprocess(text):

    # Tokenize the text

    tokens = word_tokenize(text)

    # Remove stop words and punctuation

    stop_words = set(stopwords.words('english'))

    punctuation = set(string.punctuation)

    

    filtered_tokens = [word.lower() for word in tokens if (word.isalpha() and word.lower() not in stop_words and word not in punctuation)]

    return filtered_tokens

    # Read the external file

file_path = 'SBA927.txt'

with open(file_path, 'r', encoding='utf-8') as file:

    # Read the entire content of the file

    dataset = file.read().splitlines()

    # Tokenization and Preprocessing for the entire dataset

processed_dataset = [tokenize_and_preprocess(text) for text in dataset]

# Display the results

for i, text_tokens in enumerate(processed_dataset):

    print(f"Original Text {i + 1}: {dataset[i]}")

    print(f"Processed Tokens {i + 1}: {text_tokens}")

    print("\n")