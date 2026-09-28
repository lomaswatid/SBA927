
# Import necessary libraries

import nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources if not already present

nltk.download('wordnet')

# Define a function for stemming and lemmatization

def stem_and_lemmatize(tokens):

    # Initialize stemming and lemmatization tools

    stemmer = PorterStemmer()

    lemmatizer = WordNetLemmatizer()


    # Apply stemming

    stemmed_tokens = [stemmer.stem(token) for token in tokens]


    # Apply lemmatization

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]


    return stemmed_tokens, lemmatized_tokens


# Assume 'processed_dataset' contains the preprocessed text data


# Apply stemming and lemmatization to the entire dataset

stemmed_and_lemmatized_dataset = [stem_and_lemmatize(tokens) for tokens in processed_dataset]


# Display the results

for i, (stemmed_tokens, lemmatized_tokens) in enumerate(stemmed_and_lemmatized_dataset):

    print(f"Original Tokens {i + 1}: {processed_dataset[i]}")

    print(f"Stemmed Tokens {i + 1}: {stemmed_tokens}")

    print(f"Lemmatized Tokens {i + 1}: {lemmatized_tokens}")

    print("\n")