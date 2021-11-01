#!/usr/bin/env python

"""
# Anagram Finder
Find all anagrams for text in standard input or command line argument

```bash
echo "text to be scrambled" | python anagrams/main.py

# Or

python anagrams/main.py "text to be scrambled"
```
"""

from collections import Counter
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

WORD_LOOKUP_PATH = Path('WORD_LOOKUP.pickle')


def get_words():
    """
    Get all words from a corpus, downloading if necessary
    """
    from nltk.corpus import brown

    try:
        return brown.words()
    except LookupError:
        import nltk
        nltk.download('brown')
        return brown.words()


def _examine_min_norm_freq_threshold(norm_freqs, threshold):
    """
    Look at a sample of words whose normalised frequencies are just above or just below the threshold
    """
    N = 5
    included = {k: v for k, v in norm_freqs.items() if v >= threshold}
    excluded = {k: v for k, v in norm_freqs.items() if v < threshold}
    return {
        'included': sorted(included, key=included.get, reverse=False)[:N],
        'excluded': sorted(excluded, key=excluded.get, reverse=True)[:N],
    }


def build_word_lookup(min_word_length=4):
    """
    Pre-calculate a lookup containing the letter counts in each word.
    e.g. For "that", we want {'t': 2, 'a': 1, 'h': 1}
    """

    if WORD_LOOKUP_PATH.exists():
        return pd.read_pickle(WORD_LOOKUP_PATH)

    words = get_words()

    import nltk

    freqs = nltk.FreqDist(w.lower() for w in words)
    max_freq = max(freqs.values())
    norm_freqs = {k: v / max_freq for k, v in freqs.items()}

    # Threshold found through experimentation with `_examine_min_norm_freq_threshold`
    # Make it higher if there are too many stupid/weird words
    threshold = 1e-4
    words = [k for k, v in norm_freqs.items() if v >= threshold]

    letters = [Counter(w.lower()) for w in words]
    df = pd.concat([
        pd.DataFrame({'word': words}),
        pd.DataFrame.from_records(letters).fillna(0).astype(int),
    ], axis=1).drop_duplicates()
    df['length'] = df.drop('word', axis=1).sum(axis=1)
    df = (
        df
        .loc[
            ((df['length'] > 1) | (df['word'].isin(['i', 'a'])))
            & (df['length'] >= min_word_length)
        ]
        .sort_values(['length', 'word'], ascending=False)
        .reset_index(drop=True)
    )

    df = df.drop('length', axis=1).set_index('word')
    df.to_pickle(WORD_LOOKUP_PATH)

    return df


def find_anagrams(
    text: str,
    words: pd.DataFrame = build_word_lookup(),
    stream_results: bool = True,
):
    """
    `text`: the text to be scrambled into anagrams
    `words`: a DataFrame with a row for every word in the corpus and
        a column for every character in the corpus. Each value gives the
        number of times a given character occurs in a given word.
    `stream_results`: if True, print anagrams to stdin as they're found
    """
    # A count of the characters in the input text
    text_letters = pd.Series(
        Counter(char for char in text if char in words.columns),
        index=words.columns
    ).fillna(0).astype(int)

    def _inner(
        text_letters: str,
        words: pd.DataFrame,
        all_results: set,
        current_result: Optional[tuple] = None,
    ):
        # TODO: is there an iterative implementation of this? Would that be more performant? Which would lend itself better to multiprocessing?
        """
        Main anagram algorithm:
        1. find all words that are contained in `text_letters`
        2. if none found, stop and append any `current_results` to `all_results`
        3. for each word found (if any), branch into a new function call,
            where the letters of the word are stripped from `text_letters`
            and `current_result` contains the word. Go from step 1 again
        """

        if current_result is None:
            current_result = tuple()

        contained_words = (text_letters - words).min(axis=1) >= 0

        if not contained_words.any():
            has_remainder = all(text_letters == 0)
            # TODO: checking if it's already in the results may be slow. Can I avoid this ever happening in the first place?
            if has_remainder or not current_result or current_result in all_results:
                return
            if stream_results:
                print(current_result)
            all_results.add(current_result)
            return

        # Since we're stripping out letters as we go, we can drop any words
        # that didn't match this time through
        words = words.loc[contained_words]

        for word, letters in words.iterrows():
            _inner(
                text_letters - letters,
                words,
                all_results,
                tuple(sorted((*current_result, word))),
            )

    all_results = set()

    # Recursively build anagrams from the input text's characters
    # Append these to `all_results`
    _inner(text_letters, words, all_results)

    return all_results


if __name__ == '__main__':
    text = ' '.join(sys.argv[1:])

    if text in ('help', '--help', '-h'):
        print(__doc__)
        sys.exit(0)

    if text:
        find_anagrams(text)
    else:
        for line in sys.stdin:
            find_anagrams(line)
