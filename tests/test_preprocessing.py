import pytest
from app.utils.preprocessing import clean_text

def test_clean_text():
    assert clean_text("This is a TEST!") == "test"
    assert clean_text("Hello 123") == "hello 123"
    assert clean_text("") == ""
    assert clean_text("Don't stop") == "dont stop"
    assert clean_text(123) == ""