import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk

nltk.download('wordnet')
nltk.download('punkt')

wordnet_lemmatizer = WordNetLemmatizer()


# Lemmatize a single string by tokenising and then reassembling it
def lemmatize_string(text):
    tokens = word_tokenize(text)
    output = []
    for token in tokens:
        output.append(wordnet_lemmatizer.lemmatize(token))
    return ' '.join(output)


def lemmatize(data, column):
    df = data.copy()
    df[column] = df[column].apply(lambda x: lemmatize_string(x))
    return df


def remove_dont_knows(data, column):
    df = data.copy()
    df['dontknow'] = df[column].apply(lambda x: replace_dont_knows(x, ''))
    df['dontknow'].replace('', np.nan, inplace=True)
    df.dropna(subset=['dontknow'], inplace=True)
    df.drop(columns=['dontknow'], axis=1, inplace=True)
    return df


def replace_dont_knows(text, replacement):
    terms = ['i honestly dont know', "im dont know", "i m dont know", "i dont really know", "i dont know really", "i dont know sorry", "sorry i dont know", "i dont know mate", "i dont know", "i don t know", "i do not know", "dont really know", "dont know", "i dunno", "dunno", "don t know","idk", "do not know"]
    for word in terms:
        text = text.replace(word, replacement)

    terms = ['i am not really sure', 'im not too sure', 'im not really sure', 'not really sure', "i have no idea", "i am not sure", "im not sure", "i m not sure", "im unsure", "unsure", "not sure", "not too sure", "no idea", "not a clue", "no clue"]
    for word in terms:
        text = text.replace(word, replacement)

    terms = ["no comments", "no comment","no opinion", "n a", "na", "none", "unknown", "no answer" ,'not applicable', 'nil', 'no']
    for word in terms:
        text = text.replace(word, replacement)

    terms = ['i cant think of anything', 'cant think of anything']
    for word in terms:
        text = text.replace(word, replacement)

    return text.strip()


def replace_all_domain_terms(text):
    onetoone = ['1 2 1', '1 on 1', '1 to 1', '1-1', '1on1', '1:1', '1to1']
    vle = ['blackboard', 'moodle', 'canvas']
    online = ['online learning', 'learning online',
              'learn online', 'teaching online',
              'online teaching', 'online courses', 'online course',
              'online classes'
              'online class', 'online lessons', 'online lesson',
              'online lectures', 'online lecture'
              ]
    online_meeting_tool = ['zoom', 'blackboard collaborate', 'microsoft teams', 'big blue button', 'teams']

    text = replace_domain_terms(text, onetoone, 'onetoone')
    text = replace_domain_terms(text, vle, 'vle')
    text = replace_domain_terms(text, online, 'online learning')
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
