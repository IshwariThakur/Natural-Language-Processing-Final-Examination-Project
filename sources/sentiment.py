from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_docs(docs):

    positive = 0
    neutral = 0
    negative = 0

    for doc in docs:

        score = analyzer.polarity_scores(
            doc["text"]
        )["compound"]

        if score > 0.05:
            positive += 1
        elif score < -0.05:
            negative += 1
        else:
            neutral += 1

    return positive, neutral, negative