from humourscope.preprocess import clean_text


def test_clean_basic():
    s = "Check r/funny by u/tester https://x.y ! LOL..."
    c = clean_text(s)
    assert "r/" not in c and "u/" not in c
    assert "http" not in c
