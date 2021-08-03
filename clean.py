import numpy as np


def remove_dont_knows(data, column):
    df = data.copy()
    df['dontknow'] = df[column].apply(lambda x: replace_dont_knows(x, ''))
    df['dontknow'].replace('', np.nan, inplace=True)
    df.dropna(subset=['dontknow'], inplace=True)
    df.drop(columns=['dontknow'], axis=1, inplace=True)
    return df


def replace_dont_knows(text, replacement):
    terms = ['i honestly dont know', "im dont know", "i m dont know", "i dont know", "i don t know", "i do not know", "dont know", "i dunno", "dunno", "don t know","idk", "do not know"]
    for word in terms:
        text = text.replace(word, replacement)

    terms = ["i have no idea", "i am not sure", "im not sure", "i m not sure", "unsure", "not sure", "not too sure", "no idea", "not a clue", "no clue"]
    for word in terms:
        text = text.replace(word, replacement)

    terms = ["no comments", "no comment","no opinion", "n a", "na", "none", "unknown", "no answer" ,'not applicable', 'nil', 'no']
    for word in terms:
        text = text.replace(word, replacement)

    return text.strip()


def replace_all_domain_terms(text):
    onetoone = ['1 2 1', '1 on 1', '1 to 1', '1-1', '1on1', '1:1', '1to1']
    vle = ['blackboard', 'moodle', 'canvas']
    online_meeting_tool = ['zoom', 'blackboard collaborate', 'teams', 'microsoft teams', 'big blue button']

    text = replace_domain_terms(text, onetoone, 'onetoone')
    text = replace_domain_terms(text, vle, 'vle')
    text = replace_domain_terms(text, online_meeting_tool, 'onlinemeetingtool')

    return text


def replace_domain_terms(text, domain_terms, replacement):
    for word in domain_terms:
        text = text.replace(word, replacement)
    return text


def clean(data, column=0) -> object:
    df = data.copy()

    # remove apostrophes
    df['cleaned'] = df[column].str.replace("'", '').str.replace("`",'').str.replace("’",'').str.replace("’",'')
    df['cleaned'].replace('', np.nan, inplace=True)
    df.dropna(subset=['cleaned'], inplace=True)

    # lowercase
    df['cleaned'] = df['cleaned'].str.lower()

    # Domain terms
    df['cleaned'] = df['cleaned'].apply(lambda x: replace_all_domain_terms(x))

    # remove punctuation, remove extra whitespace in string and on both sides of string
    df['cleaned'] = df['cleaned'].str.replace('[^a-z]', ' ').str.replace(' +', ' ').str.strip()

    # if the result is an empty string, or null, remove the row
    df['cleaned'].replace('', np.nan, inplace=True)
    df.dropna(subset=['cleaned'], inplace=True)

    return df
