import re
import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
#For loading lists
import os
import sys


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

wordnet_lemmatizer = WordNetLemmatizer()


def __get_wordnet_pos__(treebank_tag):
    """
    return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v)
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN


def lemmatize_string(text):
    """
    Lemmatize a single string by tokenising and then reassembling it
    :param text: the string to lemmatize
    :return: the lemmatized string
    """
    tokens = word_tokenize(text)
    pos_tokens = nltk.pos_tag(tokens)
    output = []
    for token in pos_tokens:
        output.append(wordnet_lemmatizer.lemmatize(token[0], __get_wordnet_pos__(token[1])))
    return ' '.join(output)


def lemmatize(data, columns):
    """
    Lemmatize the content of a specific column in a DataFrame
    :param data: the DataFrame
    :param columns: a single column name or list of column names
    :return: the modified DataFrame
    """
    df = data.copy()
    if type(columns) is str:
        columns = [columns]

    for column in columns:
        df[column] = df[column].apply(lambda x: lemmatize_string(x) if x is not np.nan else np.nan)
    return df


def remove_dont_knows(data, columns) -> pd.DataFrame:
    """
    Remove 'don't know' answers from a dataframe. This removes
    any rows that only contain 'don't know'. __Note__ that if multiple columns
    are supplied this will remove a row where any of the columns is 'dont know'
    :param data: the DataFrame
    :param columns: a single column name or list of column names
    :return: the modified DataFrame
    """
    df = data.copy()
    if type(columns) is str:
        columns = [columns]
    for column in columns:
        df['dontknow'] = df[column].apply(lambda x: __replace_dont_knows__(x, '') if x is not np.nan else np.nan)
        df['dontknow'].replace('', np.nan, inplace=True)
        df.dropna(subset=['dontknow'], inplace=True)
        df.drop(columns=['dontknow'], axis=1, inplace=True)
    return df


def replace_dont_knows(data, columns, replacement='') -> pd.DataFrame:
    df = data.copy()
    if type(columns) is str:
        columns = [columns]
    for column in columns:
        df[column].apply(lambda x: __replace_dont_knows__(x, replacement) if x is not np.nan else np.nan)
        df[column].replace('', np.nan, inplace=True)
    return df


def __replace_dont_knows__(text, replacement):
    """
    Replace 'dont know' or any of its synonyms in a string with the replacement string
    :param text: the string to modify
    :param replacement: the replacement string
    :return:
    """
    terms = ['i honestly dont know', "im dont know", "i m dont know", "i really dont know", "i dont really know",
             "i dont know really", "i dont know sorry", "sorry i dont know", "i dont know mate", "i dont know",
             "i don t know", "i do not know", "dont really know", "dont know", "i dunno", "dunno", "don t know", "idk",
             "do not know", 'i do nt know', 'do nt know', 'do nt know', 'i dont know to be honest', 'i do nt think there be anything',
             'i do nt even know']
    for word in terms:
        text = text.replace(word, replacement)

    terms = ['i am not really sure', 'im not too sure', 'im not really sure', 'not really sure', "i have no idea",
             "i am not sure", "im not sure", "i m not sure", "im unsure", "unsure", "not sure", "not too sure",
             "no idea", "not a clue", "no clue", 'do nt understand']
    for word in terms:
        text = text.replace(word, replacement)

    terms = ["no comments", "no comment", "no opinion", "n a", "na", "none", "unknown", "no answer", 'not applicable',
             'nil', 'no']
    for word in terms:
        text = text.replace(word, replacement)

    terms = ['i cant think of anything', 'cant think of anything', 'i ca nt say', 'ca nt think of anything', 'i ca nt think of any',
             'ca nt think of any']
    for word in terms:
        text = text.replace(word, replacement)

    return text.strip()

def replace_all_domain_terms(text, online_learning_platform_list=None, online_meeting_tool_list=None, domain_term=None, domain_string=None):
    """
    Replace any domain terms within a string
    :param text: the string to process
    :return: the processed string with domain terms normalised
    """
    onetoone = ['1 2 1', '1 on 1', '1 to 1', '1-1', '1on1', '1:1', '1to1']

    online = ['online learning', 'learning online',
              'learn online', 'teaching online',
              'online teaching', 'online courses', 'online course',
              'online classes'
              'online class', 'online lessons', 'online lesson',
              'online lectures', 'online lecture'
              ]

    if online_learning_platform_list is None:
        online_learning_platform_list = ['blackboard', 'moodle', 'canvas']
    online_learning_platform = online_learning_platform_list

    if online_meeting_tool_list is None:
        online_meeting_tool_list = ['zoom', 'blackboard collaborate', 'microsoft teams', 'ms teams', 'microsoft team', 'ms team', 'big blue button']
    online_meeting_tool = online_meeting_tool_list

    # checks both lists are same length
    if domain_term is not None:
        number_of_extra_lists = len(domain_term)
        if domain_string is not None:
            if len(domain_term) != len(domain_string):
                print("Domain terms and domain words must be the same length")
                return None
            else:
                print(number_of_extra_lists,
                      ':user specified lists were added to the replace_all_domain_terms function')

    if (domain_term is not None) and (domain_string is not None):
        for term_list, y in zip(domain_term, domain_string):
            text = replace_domain_terms(text, term_list, y)

    text = replace_domain_terms(text, onetoone, 'onetoone')
    text = replace_domain_terms(text, online_learning_platform, 'onlinelearningplatform')
    text = replace_domain_terms(text, online, 'online learning')
    text = replace_domain_terms(text, online_meeting_tool, 'onlinemeetingtool')

    return text


def replace_domain_terms(text, domain_terms, replacement):
    """
    Replace domain terms within text
    :param text: the text to process
    :param domain_terms: the list of domain terms
    :param replacement: the replacement for the domain terms
    :return: the processed string
    """
    for word in domain_terms:
        text = text.replace(word, replacement)
    return text


def __clean__(df, column, inplace, online_learning_platform_list=None, online_meeting_tool_list=None, domain_term=None, domain_string=None):

    df = df.copy()

    if not inplace:

        cleaned = 'cleaned'
        if cleaned in df.columns:
            cleaned = cleaned + "_" + column

        df[cleaned] = df[column]
        column = cleaned

    # remove apostrophes
    df[column] = df[column].apply(
        lambda x: str(x).replace("'", '').replace("`", '').replace("’", '').replace("’", '')
        if x is not np.nan else np.nan)

    # Put Teams with a capital as an onlinelearningplatform string as it can be a term for team working etc
    # Before the lowercase amendment

    df[column] = df[column].apply(
        lambda x: str(x).replace('Teams', 'onlinelearningplatform')
        if x is not np.nan else np.nan)


    # lowercase
    df[column] = df[column].apply(
        lambda x: x.lower()
        if x is not np.nan else np.nan)

    # Domain terms
    df[column] = df[column].apply(
        lambda x: replace_all_domain_terms(x, online_learning_platform_list, online_meeting_tool_list, domain_term, domain_string)
        if x is not np.nan else np.nan)

    # remove punctuation, remove extra whitespace in string and on both sides of string
    df[column] = df[column].apply(
        lambda x: re.sub(r'[^a-z]', ' ', x).replace(' +', ' ').strip()
        if x is not np.nan else np.nan)

    df[column].replace('', np.nan, inplace=True)

    return df


def clean(data, columns, inplace=False, online_learning_platform_list=None, online_meeting_tool_list=None, domain_term=None, domain_string=None) -> pd.DataFrame:
    """
    Cleans a dataframe
    :param data: the dataframe to clean
    :param columns: a single column name or list of column names to clean
    :param inplace: if True, changes are made to the specified column; otherwise, a 'cleaned'
    column is appended to the dataframe
    :return: the cleaned dataframe
    """
    df = data.copy()

    if type(columns) is str:
        columns = [columns]

    for column in columns:
        df = __clean__(df, column, inplace, online_learning_platform_list, online_meeting_tool_list, domain_term, domain_string)
    return df

    # Clean without domain for summarising apps

def __clean_without_domains__(df, column, inplace):

    df = df.copy()

    if not inplace:

        cleaned = 'cleaned'
        if cleaned in df.columns:
            cleaned = cleaned + "_" + column

        df[cleaned] = df[column]
        column = cleaned

    # remove apostrophes
    df[column] = df[column].apply(
        lambda x: str(x).replace("'", '').replace("`", '').replace("’", '').replace("’", '')
        if x is not np.nan else np.nan)

    # lowercase
    df[column] = df[column].apply(
        lambda x: x.lower()
        if x is not np.nan else np.nan)

       # remove punctuation, remove extra whitespace in string and on both sides of string
    df[column] = df[column].apply(
        lambda x: re.sub(r'[^a-z]', ' ', x).replace(' +', ' ').strip()
        if x is not np.nan else np.nan)

    df[column].replace('', np.nan, inplace=True)

    return df


def clean_without_domain(data, columns, inplace=False) -> pd.DataFrame:
    """
    Cleans a dataframe
    :param data: the dataframe to clean
    :param columns: a single column name or list of column names to clean
    :param inplace: if True, changes are made to the specified column; otherwise, a 'cleaned'
    column is appended to the dataframe
    :return: the cleaned dataframe
    """
    df = data.copy()

    if type(columns) is str:
        columns = [columns]

    for column in columns:
        df = __clean_without_domains__(df, column, inplace)
    return df

    # Functions below included for future automation of terms/lists of terms
    # For user defined list, have a folder in the directory named user-defined_corpus
    # With txt files listing what they want redacted

def bundle_dir():
    return os.path.abspath(os.path.dirname(__file__))


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if not getattr(sys, 'frozen', False):
        relative_path = 'user_defined_corpus' + os.sep + relative_path

    return os.path.join(bundle_dir(), relative_path)


def load_list(filename):
    """
    Loads a list from a file, ignoring comments
    :param filename: the name of the file to load
    :return: a list
    """
    path = resource_path(filename)
    list = []
    with open(path) as f:
        for line in f:
            if not line.strip().startswith("#") and len(line.strip()) > 0:
                list.append(line.strip())
    return list