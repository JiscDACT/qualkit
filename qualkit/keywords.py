from rake_nltk import Rake
import nltk
from nltk.stem import WordNetLemmatizer

from qualkit.stopwords import stopwords

nltk.download('wordnet')

# initiate nltk lemmatiser
lemma = WordNetLemmatizer()

r = Rake(stopwords=stopwords, min_length=1, max_length=4)


def rake_implement(x, r):
    r.extract_keywords_from_text(x)
    return r.get_ranked_phrases()


def add_keywords(df, column):
    df = df.copy()
    df['keywords'] = df[column].apply(lambda x: rake_implement(x, r))
    df['keywords'] = df['keywords'].apply(lambda x: [lemma.lemmatize(y) for y in x])
    df = df.explode('keywords')
    df.dropna(subset=['keywords'], inplace=True)
    return df

