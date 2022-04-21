import pandas as pd
import qualkit.clean
import pytest

def test_replace_dont_knows():
    text = qualkit.clean.__replace_dont_knows__('i dont know', 'idk')
    assert text == 'idk'


def test_replace_domain_terms():
    text = 'I use Blackboard'
    domain = ['Blackboard', 'Moodle']
    text = qualkit.clean.replace_domain_terms(text, domain, "a VLE")
    assert text == 'I use a VLE'


def test_lemmatize():
    text = {"text": ['more seminars running helping cooks find']}
    df = pd.DataFrame(text, columns=['text'])
    df = qualkit.clean.lemmatize(df, 'text')
    output = df['text'][0]
    assert output == 'more seminar run help cook find'


def test_lemmatize_string():
    output = qualkit.clean.lemmatize_string('more seminars running helping cooks find')
    assert output == 'more seminar run help cook find'


# More specific lemmatize tests
# Gold Standard mainly uses output taken from here: https://cental.uclouvain.be/treetagger/
def test_lem_isolate():
    #starting dataframe
    df_start = pd.DataFrame(['it is very isolating', 'i feel isolated'],
                          columns=['Q1'])
    #expected dataframe
    df_expected = pd.DataFrame(['it be very isolating', 'i feel isolate'],
                          columns=['Q1'])
    #testing the result
    assert qualkit.clean.lemmatize(df_start, 'Q1').equals(df_expected), 'isolating/isolated is not being lemmatised as standard'


def test_lem_quiet():
    #starting dataframe
    df_start = pd.DataFrame(['it is quieter working from home', 'i can quietly get on with work in my own time'],
                          columns=['Q1'])
    #expected dataframe
    df_expected = pd.DataFrame(['it be quiet work from home', 'i can quietly get on with work in my own time'],
                          columns=['Q1'])
    #testing the result
    assert qualkit.clean.lemmatize(df_start, 'Q1').equals(df_expected), 'quiet is not being lemmatised as standard'
# qualkit more eager to lemmatise than treetagger


def test_lem_convenient():
    #starting dataframe
    df_start = pd.DataFrame(['i find it more convenient working from home', 'it is just the convenience really', 'the laptop conveniently connects to the remote desktop'],
                          columns=['Q1'])
    #expected dataframe
    df_expected = pd.DataFrame(['i find it more convenient work from home', 'it be just the convenience really', 'the laptop conveniently connect to the remote desktop'],
                          columns=['Q1'])
    #testing the result
    assert qualkit.clean.lemmatize(df_start, 'Q1').equals(df_expected), 'convenient is not being lemmatised as standard'
# qualkit more eager to lemmatise than treetagger

def test_lem_collaborate():
    #starting dataframe
    df_start = pd.DataFrame(['it can be more collaborative', 'it enables more collaboration', 'my students worked collaboratively with each other'],
                          columns=['Q1'])
    #expected dataframe
    df_expected = pd.DataFrame(['it can be more collaborative', 'it enable more collaboration', 'my student work collaboratively with each other'],
                          columns=['Q1'])
    #testing the result
    assert qualkit.clean.lemmatize(df_start, 'Q1').equals(df_expected), 'collaborate is not being lemmatised as standard'


def test_lem_flexible():
    #starting dataframe
    df_start = pd.DataFrame(['i have found it more flexible', 'it allows greater flexibility'],
                          columns=['Q1'])
    #expected dataframe
    df_expected = pd.DataFrame(['i have find it more flexible', 'it allow great flexibility'],
                          columns=['Q1'])
    #testing the result
    assert qualkit.clean.lemmatize(df_start, 'Q1').equals(df_expected), 'flexible is not being lemmatised as standard'


def test_remove_dont_knows():
    text = {"text":
            [
                'i think it needs more salt',
                'i really dont know',
                'i dont know',
                'i have no idea',
                'no comment',
                'its too spicy',
                'idk'
            ]
            }
    df = pd.DataFrame(text, columns=['text'])
    df = qualkit.clean.remove_dont_knows(df, 'text')
    assert df.size == 2
    assert df['text'].iloc[0] == 'i think it needs more salt'


def test_clean():
    text = {"text": ["I'm a teapot"]}
    df = pd.DataFrame(text, columns=['text'])
    df = qualkit.clean.clean(df, 'text')
    assert df['cleaned'].iloc[0] == 'im a teapot'
    assert df['text'].iloc[0] == 'I\'m a teapot'


def test_clean_inner():
    text = {"text": ["-"]}
    df = pd.DataFrame(text)
    df = qualkit.clean.__clean__(df, 'text', inplace=False)
    assert df['text'].iloc[0] == '-'
    assert pd.isnull(df['cleaned'].iloc[0])


def test_clean_inplace():
    text = {"text": ["I'm a teapot"]}
    df = pd.DataFrame(text, columns=['text'])
    df = qualkit.clean.clean(df, 'text', inplace=True)
    assert df['text'].iloc[0] == 'im a teapot'


def test_clean_multiple_inplace():
    text = {
            "text1": ["I'm a teapot", "", "row three :)"],
            "text2": ['', '', ''],
            "text3": ['Short and stout', 'Bananas are the only fruit!', '']
            }
    df = pd.DataFrame(text)
    df = qualkit.clean.clean(df, ['text1', 'text2', 'text3'], inplace=True)
    assert df['text1'].iloc[0] == 'im a teapot'


def test_clean_multiple():
    text = {
            "text1": ["I'm a teapot", "", "row three :)"],
            "text2": ['', '', ''],
            "text3": ['Short and stout', 'Bananas are the only fruit!', '']
            }
    df = pd.DataFrame(text)
    df = qualkit.clean.clean(df, ['text1', 'text2', 'text3'])
    assert df['cleaned'].iloc[0] == 'im a teapot'
