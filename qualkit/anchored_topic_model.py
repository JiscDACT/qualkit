import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct
from qualkit.stopwords import stopwords


def anchored_topic_model(data, column, topic_names=None, anchors=None, number_of_topics=10, print_topic_details=False):

    df = data.copy()

    if anchors is not None:
        number_of_topics = len(anchors)
        if topic_names is not None:
            if len(topic_names) != len(anchors):
                print("Topic names and anchors must be the same length")
                return None
            topic_names = ['No matching topic']+topic_names

    vectorizer = TfidfVectorizer(
        max_df=0.25,
        min_df=5,
        max_features=None,
        ngram_range=(1, 3),
        norm=None,
        binary=True,
        use_idf=False,
        sublinear_tf=False,
        stop_words=stopwords
    )
    vectorizer = vectorizer.fit(df[column])
    tfidf = vectorizer.transform(df[column])
    vocab = vectorizer.get_feature_names()

    # Filter by vocab
    if anchors is not None:
        anchors = [
            [a for a in topic if a in vocab]
            for topic in anchors
        ]

    # Create the Corex model
    model = ct.Corex(n_hidden=number_of_topics, seed=42)
    if anchors is None:
        model = model.fit(
            tfidf,
            words=vocab
        )
    else:
        model = model.fit(
            tfidf,
            words=vocab,
            anchors=anchors,
            anchor_strength=2  # Tells the model how much it should rely on the anchors
        )

    # Enumerate the topics from the model and create labels
    topic_labels = ['No matching topic']
    if print_topic_details:
        print("Total correlation: " + str(model.tc))
    for i, topic_ngrams in enumerate(model.get_topics(n_words=10)):
        topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
        topic_label = ", ".join(topic_ngrams)
        topic_labels.append(topic_label)
        if print_topic_details:
            print("Topic #{}: {}".format(i + 1, topic_label))
    if topic_names is None:
        topic_names = topic_labels

    # Apply the model to the documents
    doc_topics = model.transform(tfidf)
    topic_matrix = pd.DataFrame(
        doc_topics,
        columns=["topic_{}".format(i) for i in range(1, number_of_topics + 1)],
        index=df.index,
    )

    topic_cols = ["topic_" + str(i) for i in range(1, number_of_topics + 1)]
    df_document_topic = pd.DataFrame(topic_matrix, columns=topic_cols)
    df_document_topic.insert(0, 'no topic', False, True)

    # get the dominant topic for each document
    dominant_topic = (np.argmax(df_document_topic.values, axis=1))
    df_document_topic['Dominant_topic'] = dominant_topic
    df_document_topic['Topic label'] = df_document_topic['Dominant_topic'].apply(lambda x: topic_labels[x])
    df_document_topic['Topic name'] = df_document_topic['Dominant_topic'].apply(lambda x: topic_names[x])

    # remove the individual columns for topics
    df_document_topic.drop(columns=topic_cols, inplace=True, axis=1)
    topic_matrix.drop(columns=topic_cols, inplace=True, axis=1)

    # join to original dataframes
    topic_matrix = pd.merge(topic_matrix, df_document_topic[['Topic label', 'Topic name']],
                            left_index=True, right_index=True, how='outer')
    results = pd.merge(df, topic_matrix, left_index=True, right_index=True, how='outer')

    return results


