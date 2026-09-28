# Import necessary libraries

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk import pos_tag, ne_chunk


# Download required resources

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('maxent_ne_chunker')

nltk.download('words')



# Function to tokenize text into sentences

def tokenize_into_sentences(text):

    sentences = sent_tokenize(text)

    return sentences


# Function to perform part-of-speech tagging on sentences

def part_of_speech_tagging(sentences):

    pos_tagged_sentences = [pos_tag(word_tokenize(sentence)) for sentence in sentences]

    return pos_tagged_sentences


# Function to extract named entities from POS-tagged sentences

def extract_named_entities(pos_tagged_sentences):

    named_entities = []

    for sentence in pos_tagged_sentences:

        tree = ne_chunk(sentence)

        for subtree in tree:

            if hasattr(subtree, 'label'):  # Check if subtree is a named entity

                entity = " ".join([word for word, tag in subtree.leaves()])

                named_entities.append((entity, subtree.label()))

    return named_entities



# Define a function for Named Entity Recognition

def named_entity_recognition(text):

  # Tokenize and process the text

  sentences = tokenize_into_sentences(text)

  pos_tags = part_of_speech_tagging(sentences)

  named_entities = extract_named_entities(pos_tags)


  return named_entities



# Apply Named Entity Recognition to the entire dataset

ner_results_dataset = [named_entity_recognition(text) for text in dataset]


# Display the results

for i, ner_results in enumerate(ner_results_dataset):

  print(f"Named Entities in Text {i + 1}: {ner_results}")

  print("\n")