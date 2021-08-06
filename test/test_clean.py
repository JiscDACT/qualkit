import pandas as pd
import qualkit.clean


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
