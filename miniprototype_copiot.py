from transformers import pipeline

# Load Hugging Face models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")


def summarize_and_analyze(conversation_text):
    # --- Step 1: Summarize conversation ---
    summary = summarizer(conversation_text, max_length=60,
                         min_length=20, do_sample=False)[0]['summary_text']

    # --- Step 2: Sentiment analysis (based on customer side) ---
    sentiment_result = sentiment_analyzer(conversation_text)[0]
    label = sentiment_result['label']
    score = sentiment_result['score']

    # Map to simpler categories with emoji
    if label == "POSITIVE":
        sentiment = "Positive 🙂" if score < 0.85 else "Very Positive 🤩"
    elif label == "NEGATIVE":
        sentiment = "Negative 😕" if score < 0.85 else "Very Negative 😡"
    else:
        sentiment = "Neutral 😐"

    # --- Step 3: Return combined result ---
    return {
        "summary": summary,
        "sentiment": sentiment
    }


# ---- Example ----
conversation = """
Customer: I have been waiting for my refund for 2 weeks. This is really frustrating.
Agent: I understand your concern. Let me check the status.
Agent: The refund has been processed today and should reflect in your account within 2-3 business days.
Customer: Okay, I’ll wait, but this delay has been disappointing.
"""

result = summarize_and_analyze(conversation)
print("Conversation Summary:", result["summary"])
print("Sentiment:", result["sentiment"])
