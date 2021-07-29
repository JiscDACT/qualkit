import pandas as pd
from clean import clean
from topics import lda, lda_with_keywords

# read data
data = pd.read_csv('data/feedback.csv', header = None)
data = clean(data)

df = lda(data, num_topics=7)
df.to_csv('output/lda.csv')

df = lda_with_keywords(df)
df.to_csv('output/keywords.csv')

df = lda(data, num_topics=7, output='features')
df.to_csv('output/test.csv')

