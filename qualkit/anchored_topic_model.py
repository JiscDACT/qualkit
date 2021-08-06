import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from corextopic import corextopic as ct
from qualkit.stopwords import stopwords


def load_topics(file):
    """
    Loads topic names and anchors from a file
    :param file: the path/filename of the file to use
    :return: a DataFrame with 'topic_name' and 'anchors' columns
    """
    df = pd.read_csv(file, header=None)
    df['anchors'] = df.drop(0, axis=1).values.tolist()
    df['anchors'] = df['anchors'].apply(lambda x: [i for i in x if i == i])
    df['topic_name'] = df[0]
    output = df[['topic_name', 'anchors']].copy()
    return output


def anchored_topic_model(data, column, topic_filename=None, topic_names=None, anchors=None, number_of_topics=10,
                         print_topic_details=False):
    """
    Creates a topic model using the Corex algorithm using an optional set of user-provided anchors
    :param data: a DataFrame containing a column with text to analyse
    :param column: the name of the column containing the text
    :param topic_filename: (optional) the name of a file containing anchor terms and topic names
    :param topic_names: (optional) a List of topic names
    :param anchors: (optional) a list containing lists of anchor terms
    :param number_of_topics: defaults to 10; the number of topics to generate (overridden if topics are supplied)
    :param print_topic_details: if true, print to console a summary of the generated topics
    :return:a DataFrame containing the original data supplied along with the 'topic label' and 'topic name' for each row
    """

    # Load topics from file if provided
    if topic_filename is not None:
        tf = load_topics(topic_filename)
        topic_names = tf['topic_name']
        anchors = tf['anchors']

    df = data.copy()

    # Calculate number of topics based on provided anchors if supplied
    if anchors is not None:
        number_of_topics = len(anchors)
        if topic_names is not None:
            if len(topic_names) != len(anchors):
                print("Topic names and anchors must be the same length")
                return None
            topic_names = ['No matching topic']+topic_names

    # Initialise the vectorizer that will split the text into tokens
    # uses the TFIDF algorithm
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

    # Get rid of any NaNs before we run the vectorizer
    df.dropna(subset=[column], inplace=True)

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

    # Add columns for each topic
    topic_cols = ["topic_" + str(i) for i in range(1, number_of_topics + 1)]
    df_document_topic = pd.DataFrame(topic_matrix, columns=topic_cols)
    df_document_topic.insert(0, 'no topic', False, True)

    # get the dominant topic for each document, or 'no topic' if unmatched
    dominant_topic = (np.argmax(df_document_topic.values, axis=1))
    df_document_topic['Dominant_topic'] = dominant_topic
    df_document_topic['Topic label'] = df_document_topic['Dominant_topic'].apply(lambda x: topic_labels[x])
    df_document_topic['Topic name'] = df_document_topic['Dominant_topic'].apply(lambda x: topic_names[x])

    # remove the individual columns for topics
    df_document_topic.drop(columns=topic_cols, inplace=True, axis=1)
    topic_matrix.drop(columns=topic_cols, inplace=True, axis=1)

    # join the results back to the original dataframe and return
    topic_matrix = pd.merge(topic_matrix, df_document_topic[['Topic label', 'Topic name']],
                            left_index=True, right_index=True, how='outer')
    results = pd.merge(df, topic_matrix, left_index=True, right_index=True, how='outer')

    return results
