import pandas as pd
from clean import clean
from topics import lda, lda_with_keywords
from sentiment import add_sentiment_score

# 1. LDA

# read data
# data = pd.read_csv('data/feedback.csv', header = None)
# data = clean(data)
# print(data.columns.values.tolist())
# df = lda(data, num_topics=7)
# df.to_csv('output/lda.csv')
#
# df = lda_with_keywords(df)
# df.to_csv('output/keywords.csv')
#
# df = lda(data, num_topics=7, output='features')
# df.to_csv('output/test.csv')

# 2 sentiments

# data = pd.read_csv('data/dei_subset.csv')
# data = clean(data, 'Positive')
# data = add_sentiment_score(data, 'Positive')
# data['positive_score'] = data['sentiment']
# data = clean(data, 'Negative')
# data = add_sentiment_score(data, 'Negative')
# data['negative_score'] = data['sentiment']
# data.to_csv('output/sentiment.csv')

# 3 mix metadata with tokenisation and/or keyword extraction "who mentions what"
data = pd.read_csv('data/dei_student_all.csv')
data['Unique Response Number'] = data['Unique Response Number'].astype(str)

df = clean(data, 'Q26')
df = lda(df, num_topics=10)
df.rename(columns={
    "Dominant_topic": "Q26 dominant topic",
    "Topic_keywords": "Q26 topic keywords"
}, inplace=True)
df.drop(columns=['Topic_number', 'Topic1','Topic2','Topic3','Topic4','Topic5','Topic6','Topic7','Topic8','Topic9','Topic10', 'cleaned', 'tokens'])
df.dropna(subset=['Unique Response Number'], inplace=True)

df2 = clean(data, 'Q18')
df2 = lda(df2, num_topics=10)
df2.rename(columns={
    "Dominant_topic": "Q18 dominant topic",
    "Topic_keywords": "Q18 topic keywords"
}, inplace=True)
df2 = df2[['Unique Response Number','Q18 dominant topic', 'Q18 topic keywords']].copy()
df2.dropna(subset=['Unique Response Number'], inplace=True)

df['Unique Response Number'] = df['Unique Response Number'].astype(str)
df2['Unique Response Number'] = df2['Unique Response Number'].astype(str)
df3 = pd.merge(df, df2, on=['Unique Response Number'], how='inner')

df3.to_csv('output/lda_2.csv')