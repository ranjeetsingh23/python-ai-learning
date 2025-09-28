# upgraded_sentiment_checker_txt.py

from transformers import pipeline
import pandas as pd

# 1️⃣ Load text file
file_path = "sentiment_checker.txt"  # Replace with your file path
with open(file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# 2️⃣ Initialize Hugging Face sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# 3️⃣ Function to get sentiment and confidence


def get_sentiment(text):
    result = classifier(text)[0]
    return result['label'], round(result['score'], 2)


# 4️⃣ Apply sentiment analysis to each line
results = [get_sentiment(text) for text in lines]

# 5️⃣ Create a DataFrame and save results
df = pd.DataFrame({
    "Text": lines,
    "Sentiment": [r[0] for r in results],
    "Confidence": [r[1] for r in results]
})

output_file = "sentiment_results.txt"
df.to_csv(output_file, index=False)
print(f"✅ Sentiment analysis completed! Results saved to {output_file}")
