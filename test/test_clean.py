import pytest
from qualkit.clean import replace_dont_knows, replace_domain_terms


def test_replace_dont_knows():
    text = replace_dont_knows('i dont know', 'idk')
    assert text == 'idk'


def test_replace_domain_terms():
    text = 'I use Blackboard'
    domain = ['Blackboard', 'Moodle']
    text = replace_domain_terms(text, domain, "a VLE")
    assert text == 'I use a VLE'