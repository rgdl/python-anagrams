from anagrams.main import find_anagrams


def test_all_text_is_an_anagram_of_its_own_anagrams():
    test_text = 'good anagrams'
    results = find_anagrams(test_text, stream_results=False)
    assert ('anagrams', 'good') in results


def test_being_an_anagram_is_an_equivalence_relation():
    test_text = 'good anagrams'
    results = find_anagrams(test_text, stream_results=False)
    for result in results:
        new_results = find_anagrams(' '.join(result))
        assert new_results == results

