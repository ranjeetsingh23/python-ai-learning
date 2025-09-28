from transformers import pipeline

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to map score + label to 5-level sentiment and emoji


def sentiment_with_emoji(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    score = result['score']

    if label == "POSITIVE":
        if score > 0.85:
            sentiment = "Very Positive"
            emoji = "🤩"
        else:
            sentiment = "Positive"
            emoji = "🙂"
    elif label == "NEGATIVE":
        if score > 0.85:
            sentiment = "Very Negative"
            emoji = "😡"
        else:
            sentiment = "Negative"
            emoji = "😕"
    else:
        sentiment = "Neutral"
        emoji = "😐"

    return f"{text} → {sentiment} {emoji}"


# Example texts
texts = [
    "I love this product!",
    "This is okay, nothing special.",
    "I hate waiting in long lines!"
]

for t in texts:
    print(sentiment_with_emoji(t))
