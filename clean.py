import numpy as np


def replace_domain_terms(text):
    vle = ['blackboard', 'moodle', 'canvas']
    online_meeting_tool = ['zoom', 'blackboard collaborate', 'teams', 'microsoft teams', 'big blue button']
    for word in vle:
        text = text.replace(word, 'vle')
    for word in online_meeting_tool:
        text = text.replace(word, 'online_meeting_tool')
    return text


def clean(data):
    df = data.copy()
    df.dropna(inplace=True)

    # case text as lowercase, remove punctuation, remove extra whitespace in string and on both sides of string
    df['cleaned'] = df[0].str.lower().str.replace('[^a-z]', ' ').str.replace(' +', ' ').str.strip()
    df['cleaned'].replace('', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Domain synonyms
    df['cleaned'] = df['cleaned'].apply(lambda x: replace_domain_terms(x))

    return df