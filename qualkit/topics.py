import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import collections

from qualkit.stopwords import stopwords

nltk.download('wordnet')

# The code for combining LDA and RAKE is based upon Lowri Williams' method described here:
# https://github.com/LowriWilliams/Topic_Modelling_Beyond_Tokens/

# initiate stopwords
stop_words = stopwords

# initiate nltk lemmatiser
wordnet_lemmatizer = WordNetLemmatizer()


def convert_to_tokens(data):
    df = data.copy()

    # tokenise string
    df['tokens'] = df['cleaned'].apply(lambda x: nltk.word_tokenize(x))

    # remove stopwords
    df['tokens'] = df['tokens'].apply(lambda x: [item for item in x if item not in stop_words])

    # lemmatise words
    df['tokens'] = df['tokens'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])

    # Remove nulls
    df['tokens'].replace('', np.nan, inplace=True)
    df.dropna(subset=['tokens'], inplace=True)

    return df


def create_lda_model(num_topics=10):
    # initisalise LDA Model
    lda_model = LatentDirichletAllocation(n_components=num_topics, # number of topics
                                      random_state=10,          # random state
                                      evaluate_every=-1,      # compute perplexity every n iters, default: Don't
                                      n_jobs=-1,              # Use all available CPUs
                                     )

    return lda_model


# Remove duplicates from the keywords extracted from the topic modelling output.
# This is because we want to limit the amount of Ideas that go across multiple topics.
# We want to stick to 1 idea to 1 topic
def remove_duplicates(topic_keywords):

    dupes = []

    for i in topic_keywords:
        for j in i:
            dupes.append(str(j))

    dupes = [item for item, count in collections.Counter(dupes).items() if count > 1]

    topic_keywords_processed = []

    for i in topic_keywords:
        tmp = []
        for j in i:
            if str(j) not in dupes:
                tmp.append(str(j))
            else:
                pass
        topic_keywords_processed.append(tmp)

    return topic_keywords_processed


def add_keywords_to_topics(data):
    df = data.copy()
    all_results = []

    for i in df['Dominant_topic'].unique():
        topic = df[df['Dominant_topic'] == i]
        topic = topic.copy()

        key_words = []
        topic.dropna(inplace=True)

        # run keyword extraction
        for j in topic['cleaned'].values.tolist():
            r = Rake(stopwords=stop_words)

            a = r.extract_keywords_from_text(j)
            c = r.get_ranked_phrases_with_scores()

            for k in c:
                if k not in key_words:
                    key_words.append(k)

        key_words = pd.DataFrame(key_words, columns=['score', 'term'])
        key_words = key_words.sort_values('score', ascending=False)
        key_words = key_words.drop_duplicates(subset=['term'])
        key_words['topic_number'] = i
        key_words['term_list'] =  key_words.term.apply(lambda x: x.split())

        # find bigrams from key words to match against topic modelling output
        tmp_keywords = []
        for j in key_words.values.tolist():
            tmp = []
            bi_grams = ngrams(j[3], 2)
            for g in bi_grams:
                tmp.append(' '.join(g))
            for k in j[3]:
                # lemmatise words to match the lemmatised output of the topic modelling word extraction
                tmp.append(wordnet_lemmatizer.lemmatize(k))
            j.remove(j[3])
            j.append(list(set(tmp)))
            tmp_keywords.append(j)

        # mask key words against topic modelling output
        key_words = pd.DataFrame(tmp_keywords, columns=['score', 'term', 'topic_number', 'term_list'])
        topic_keywords = topic['Topic_keywords'].values.tolist()
        topic_keywords = [item for sublist in topic_keywords for item in sublist]
        topic_keywords = list(set(topic_keywords))

        tmp = []

        for t in topic_keywords:

            mask = key_words.term_list.apply(lambda x: t in x)
            key_words_processed = key_words[mask]

            if key_words_processed.empty:
                pass
            else:
                for j in key_words_processed[['score', 'term', 'topic_number', 'term_list']].values.tolist():
                    if j not in tmp:
                        tmp.append(j)

        key_words = pd.DataFrame(tmp, columns=['score', 'term', 'topic_number', 'term_list'])

        # select the max score as the topic title
        top_key_words = key_words[key_words.score == key_words['score'].max()]

        # select the remaining keywords as child terms
        remaining_keywords = key_words[key_words.score != key_words['score'].max()]

        # if there are more than 1 keyword in the topic title, aggregate them with a / as a separatore
        top_key_words = top_key_words.copy()
        top_key_words = top_key_words.groupby(['score', 'topic_number']).agg({'term' : lambda x: ' / '.join(map(str, x))})
        top_key_words = top_key_words.reset_index()
        top_key_words['parent'] = ''

        # add 0.1 to the child keywords to identify then in the merged dataframe
        remaining_keywords = remaining_keywords.copy()
        remaining_keywords['topic_number'] = remaining_keywords['topic_number'] + 0.1
        remaining_keywords['parent'] = top_key_words['term'].values.tolist()[0]

        all_topics = pd.concat([top_key_words, remaining_keywords], sort=False)

        for t in all_topics.to_dict(orient='records'):
            all_results.append(t)

    return all_results


#
# Extracts the features from the trained model along with
# their weights
#
def lda_features(lda_model, vectorizer):
    top_feature_data = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]
        top_features = [vectorizer.get_feature_names()[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        for i in range(0, topic_idx+1):
            top_feature_data.append(
                {
                    "topic": topic_idx+1,
                    "features": top_features[i],
                    "weights": weights[i]
                }
            )
    return pd.DataFrame(top_feature_data)


def lda(data, sentiment='positive', num_topics=12, output='default'):
    df = data.copy()
    df = convert_to_tokens(df)

    # Build the model
    lda_model = create_lda_model(num_topics=num_topics)

    # initialise the count vectorizer
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))

    # join the processed data to be vectorised
    vectors = []
    for index, row in df.iterrows():
        vectors.append(", ".join(row['tokens']))
    vectorised = vectorizer.fit_transform(vectors)

    # Apply model to vectorised tokens
    lda_output = lda_model.fit_transform(vectorised)

    if output == 'features':
        return lda_features(lda_model, vectorizer)

    # Add column names
    topic_names = ["Topic" + str(i) for i in range(1, lda_model.n_components + 1)]

    # make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns = topic_names)

    # get dominant topic for each document
    dominant_topic = (np.argmax(df_document_topic.values, axis=1)+1)
    df_document_topic['Dominant_topic'] = dominant_topic

    # join to original dataframes
    df = pd.merge(df, df_document_topic, left_index = True, right_index = True, how = 'outer')

    keywords = np.array(vectorizer.get_feature_names())

    topic_keywords = []

    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:20]
        topic_keywords.append(keywords.take(top_keyword_locs))

    topic_keywords = remove_duplicates(topic_keywords)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Term '+ str(i) for i in range(1, df_topic_keywords.shape[1] + 1)]
    df_topic_keywords['Topic_keywords'] = df_topic_keywords.values.tolist()
    df_topic_keywords['Topic_number'] = df_topic_keywords.index + 1
    df_topic_keywords = df_topic_keywords[['Topic_keywords', 'Topic_number']]

    # Remove None from lists
    tmp = []

    for i in df_topic_keywords['Topic_keywords']:
        tmp.append([x for x in i if x is not None])

    df_topic_keywords['Topic_keywords'] = tmp

    # Merge key terms back to main frame
    df = pd.merge(df, df_topic_keywords, left_on='Dominant_topic', right_on='Topic_number')

    return df


def lda_with_keywords(data):
    df = data.copy()
    # Add keywords to topics
    all_results = add_keywords_to_topics(df)

    all_topics_df = pd.DataFrame(all_results)
    all_topics_df = all_topics_df.sort_values('topic_number', ascending=True)

    return all_topics_df