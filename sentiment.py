import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


def add_sentiment_score(data, column, filter=None):
    data = data.copy()
    sia = SentimentIntensityAnalyzer()
    data['sentiment'] = data[column].apply(lambda x: sia.polarity_scores(x)['compound'])
    if filter == 'positive':
        data.drop(data.index[data['sentiment'] < 0.3], inplace=True)
    elif filter == 'negative':
        data.drop(data.index[data['sentiment'] > 0], inplace=True)
    elif filter == 'neutral':
        data.drop(data.index[data['sentiment'] > 0.3], inplace=True)
        data.drop(data.index[data['sentiment'] < 0], inplace=True)
    else:
        pass
    return data
