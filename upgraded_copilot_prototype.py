from transformers import pipeline
import pandas as pd
import random

# Load Hugging Face models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Emoji mapping for star ratings
emoji_map = {
    "1 star": "Very Negative 😡",
    "2 stars": "Negative 😕",
    "3 stars": "Neutral 😐",
    "4 stars": "Positive 🙂",
    "5 stars": "Very Positive 🤩",
}

# Function to summarize and analyze sentiment safely


def summarize_and_analyze(conversation_text):
    if not conversation_text.strip():  # skip empty conversations
        return {"summary": "No content", "sentiment": "Neutral 😐"}

    # --- Summarize conversation ---
    try:
        summary = summarizer(
            conversation_text, max_length=60, min_length=25, do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        summary = "Summary error"

    # --- Sentiment analysis ---
    try:
        sentiment_results = sentiment_analyzer(conversation_text)
        if sentiment_results:
            label = sentiment_results[0]["label"]
        else:
            label = "3 stars"  # default to neutral
    except Exception as e:
        label = "3 stars"

    sentiment_emoji = emoji_map.get(label, "Neutral 😐")

    return {"summary": summary, "sentiment": sentiment_emoji}


# --- Step 1: Define 30+ diverse conversation templates ---
conversation_templates = [
    """Customer: I have been waiting for my refund for 2 weeks. This is really frustrating.
Agent: I understand your concern. Let me check the status.
Agent: The refund has been processed today and should reflect in your account within 2-3 business days.
Customer: Okay, I’ll wait, but this delay has been disappointing.""",
    """Customer: The product I received is damaged and not working.
Agent: I apologize for the inconvenience. We can replace it immediately.
Customer: Thank you, I appreciate the quick response.
Agent: You should receive the replacement within 5 business days.""",
    """Customer: I was charged twice for my order. Can you fix this?
Agent: Sorry for the error. I’ve initiated a refund for the extra charge.
Customer: That’s great, thanks for the quick help!
Agent: You’re welcome! The refund will reflect in a few days.""",
    """Customer: Your app keeps crashing whenever I try to place an order.
Agent: I’m sorry about that. Have you tried reinstalling the app?
Customer: Yes, but the problem persists.
Agent: We are escalating this issue to our technical team immediately.""",
    """Customer: I love the new features in the latest update!
Agent: That’s wonderful to hear! We appreciate your feedback.
Customer: The app feels much smoother now.
Agent: We’re glad it’s working well for you!""",
    """Customer: I had trouble with billing. I was charged twice.
Agent: I apologize for the error. I’ve initiated a refund for the extra charge.
Customer: Thank you, I appreciate your help.
Agent: The refund will reflect in a few business days.""",
    """Customer: I am frustrated because my order is incomplete.
Agent: I’m sorry about that. I will arrange the missing items to be sent immediately.
Customer: Thank you, that solves it.
Agent: You’re welcome!""",
    """Customer: The product quality was not what I expected.
Agent: Sorry for the disappointment. We can offer a replacement or refund.
Customer: I’ll go with the replacement.
Agent: It will be sent out today.""",
    """Customer: I had a great experience with the support agent.
Agent: Thank you! We’re happy to help.
Customer: Made my day easier.
Agent: That’s our goal!""",
    """Customer: I can’t log into my account. Keeps saying invalid password.
Agent: Let’s reset your password. I’ll send the link to your email.
Customer: Got it. Thanks!
Agent: You should be able to log in now.""",
    """Customer: The package I received was missing items.
Agent: Apologies. I will send the missing items immediately.
Customer: Thank you, I was worried.
Agent: You should receive them in 3 days.""",
    """Customer: Customer support took too long to respond.
Agent: Sorry for the delay. We will make sure this doesn’t happen again.
Customer: Okay, thanks for acknowledging.
Agent: We appreciate your patience.""",
    """Customer: I love the quick response from your support team.
Agent: Thank you! We strive to resolve issues promptly.
Customer: Makes me trust your company more.
Agent: We’re glad to hear that!""",
    """Customer: I got the wrong size for my order.
Agent: Apologies. We can replace it with the correct size.
Customer: Great, thank you.
Agent: The correct size will be sent today.""",
    """Customer: I need help tracking my package.
Agent: Sure! Can you provide the order ID?
Customer: Yes, it’s 12345.
Agent: Your package will arrive tomorrow.""",
    """Customer: The delivery was late again!
Agent: I apologize. We are expediting it, and it should arrive tomorrow.
Customer: Okay, thanks for fixing it quickly.
Agent: You’re welcome!""",
    """Customer: I’m very happy with the product quality!
Agent: That’s wonderful to hear!
Customer: It exceeded my expectations.
Agent: We’re glad you enjoyed it!""",
    """Customer: The product color doesn’t match the image online.
Agent: Sorry about that. We can send a replacement.
Customer: That would be helpful, thanks.
Agent: Replacement will be shipped today.""",
    """Customer: The billing page is confusing.
Agent: We can walk you through it step by step.
Customer: Thanks, that helped a lot.
Agent: Glad to assist!""",
    """Customer: I need to cancel my subscription.
Agent: Sure, I can do that for you.
Customer: Thank you.
Agent: Your subscription is now canceled.""",
    """Customer: I received an expired product.
Agent: I apologize. We’ll send a fresh replacement immediately.
Customer: Thanks, I appreciate it.
Agent: You should get it within 2 days.""",
    """Customer: The app keeps logging me out.
Agent: Sorry for the trouble. Try clearing cache and re-login.
Customer: That worked, thanks!
Agent: Glad it helped!""",
    """Customer: I want to upgrade my plan.
Agent: I can help with that. Which plan would you like?
Customer: The premium plan.
Agent: Done! You have been upgraded.""",
    """Customer: Your website is slow.
Agent: Sorry for the inconvenience. We’re optimizing performance.
Customer: Hope it improves soon.
Agent: It will, thank you for your patience.""",
    """Customer: The agent was rude.
Agent: We apologize. This is not our standard. Can I assist you further?
Customer: Thank you.
Agent: We’ll make sure it doesn’t happen again.""",
    """Customer: I received the wrong address confirmation.
Agent: Sorry, we’ll correct it immediately.
Customer: Thanks.
Agent: Corrected and confirmation sent.""",
    """Customer: I love the packaging of the product.
Agent: Thank you! We’re glad you liked it.
Customer: Makes unboxing fun.
Agent: That’s our goal!""",
    """Customer: My account got blocked unexpectedly.
Agent: Sorry! We’ll unlock it immediately.
Customer: Thanks a lot.
Agent: You can now access your account.""",
    """Customer: The tracking number doesn’t work.
Agent: Apologies. I’ll provide the correct tracking number.
Customer: Got it. Thanks.
Agent: You’re welcome!"""
]

# --- Step 2: Generate 100 diverse conversations ---
conversations = [random.choice(conversation_templates) for _ in range(100)]

# --- Step 3: Analyze conversations safely ---
results = []
for idx, conv in enumerate(conversations, start=1):
    try:
        analyzed = summarize_and_analyze(conv)
        results.append({
            "ID": idx,  # Add an ID column
            "conversation": conv,
            "summary": analyzed["summary"],
            "sentiment": analyzed["sentiment"]
        })
    except Exception as e:
        print(f"Skipped conversation {idx} due to error: {e}")

# --- Step 4: Save results to a text file with emojis ---
with open("conversation_summary_100_indexed.txt", "w", encoding="utf-8") as f:
    for idx, conv_data in enumerate(results, start=1):
        f.write(f"ID: {idx}\n")
        f.write(f"Conversation:\n{conv_data['conversation']}\n")
        f.write(f"Summary: {conv_data['summary']}\n")
        f.write(f"Sentiment: {conv_data['sentiment']}\n")
        f.write("-" * 50 + "\n")

print("✅ 100 conversations saved in conversation_summary_100_indexed.txt with emojis")
