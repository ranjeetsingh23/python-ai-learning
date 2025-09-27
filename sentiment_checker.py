from textblob import TextBlob

# Function to check sentiment


def check_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😢"
    else:
        return "Neutral 😐"


# Example usage
if __name__ == "__main__":
    print("Welcome to Sentiment Checker!")
    while True:
        text = input("\nEnter a sentence (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        print("Sentiment:", check_sentiment(text))
