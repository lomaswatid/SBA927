
import os
from dotenv import load_dotenv
from groq import Groq

# Load the API key from the .env file
load_dotenv()

# Create the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Function to perform sentiment analysis
def analyze_sentiment(text):

    # Create the prompt
    prompt = f"""
    Analyze the sentiment of the following business-related text.

    Classify the sentiment as:
    - Positive
    - Negative
    - Neutral

    Explain briefly why you selected that sentiment.

    Text:
    "{text}"

    Return your answer in this format:
    Sentiment: [Positive, Negative, or Neutral]
    Explanation: [brief explanation]
    """

    # Send the prompt to the language model
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    # Return the model's response
    return response.choices[0].message.content


# Sample business-related sentences
sample_sentences = [
    "The customer service was excellent and my issue was resolved quickly.",
    "The product arrived damaged and customer support never responded.",
    "The package was delivered on Tuesday.",
    "The product works very well, but the delivery took much longer than expected."
]