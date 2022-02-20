import re
import threading
import time

import pytest

from anagrams.main import find_anagrams


@pytest.mark.parametrize('test_input,expected', [
    # tests from here: https://www.thewordfinder.com/funny-anagrams.php
    # excluded those that contained words not in my lookup table, or took too long
    ("admirer", "married"),
    ("alec guinness", "genuine class"),
    ("dormitory", "dirty room"),
    ("elvis", "lives"),
    ("evangelist", "evil's agent"),
    ("listen", "silent"),
    ("mother-in-law", "woman hitler"),
    ("postmaster", "stamp store"),
    ("sycophant", "acts phony"),
    ("the detectives", "detect thieves"),
    ("the hilton", "hint: hotel"),
    ("the morse code", "here come dots"),
])
def test_famous_examples_can_be_found(test_input: str, expected: str):
    TIMEOUT_SECONDS = 10

    results = set()

    def _target(results: set):
        """
        Perform calculation in a parallel thread
        If taking too long, throw an error
        """
        results |= find_anagrams(test_input, stream_results=False)

    thread = threading.Thread(target=_target, args=(results,))
    thread.start()
    t0 = time.time()
    while True:
        if not thread.is_alive():
            break
        if time.time() - t0 > TIMEOUT_SECONDS:
            raise RuntimeError('Took too long to calculate')

    clean_expected = re.sub('[^a-z ]', '', expected.lower())
    clean_expected_words = tuple(sorted(
        clean_expected.split(),
        key=(lambda x: (len(x), x)),
    ))
    assert clean_expected_words in results


def test_being_an_anagram_is_an_equivalence_relation():
    test_text = 'good anagrams'
    results = find_anagrams(test_text, stream_results=False)
    for result in results:
        new_results = find_anagrams(' '.join(result))
        assert new_results == results
