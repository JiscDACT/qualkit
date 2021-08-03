import pytest
import pandas as pd
from qualkit.clean import replace_dont_knows, replace_domain_terms, lemmatize, lemmatize_string


def test_replace_dont_knows():
    text = replace_dont_knows('i dont know', 'idk')
    assert text == 'idk'


def test_replace_domain_terms():
    text = 'I use Blackboard'
    domain = ['Blackboard', 'Moodle']
    text = replace_domain_terms(text, domain, "a VLE")
    assert text == 'I use a VLE'


def test_lemmatize():
    text = {"text": ['more seminars running helping cooks find']}
    df = pd.DataFrame(text, columns=['text'])
    df = lemmatize(df, 'text')
    output = df['text'][0]
    assert output == 'more seminar running helping cook find'


def test_lemmatize_string():
    output = lemmatize_string('more seminars running helping cooks find')
    assert output == 'more seminar running helping cook find'
