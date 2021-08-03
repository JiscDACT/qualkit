import pytest
from qualkit.clean import replace_dont_knows


def test_replace_dont_knows():
    text = replace_dont_knows('i dont know', 'idk')
    assert text == 'idk'