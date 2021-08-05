from qualkit import anchored_topic_model


def test_load_topics():
    df = anchored_topic_model.load_topics('test/topics.csv')
    assert df['topic_name'].size == 3
    assert df['anchors'].loc[0] == ['apple', 'banana']
    assert df['anchors'].loc[2] == ['potato', 'carrot', 'onion']