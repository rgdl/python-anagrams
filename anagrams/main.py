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
from collections import deque
from pathlib import Path
import string
import sys
from typing import Optional

import numpy as np
import pandas as pd

WORD_LOOKUP_PATH = Path('WORD_LOOKUP.pickle')
CHARACTERS_OF_INTEREST = {*string.ascii_lowercase}


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


def build_word_lookup(min_word_length: int = 4, characters: set = CHARACTERS_OF_INTEREST):
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

    # Filter
    character_cols_to_drop = [c for c in df if c not in {*characters, 'length', 'word'}]
    rows_to_drop = df[character_cols_to_drop].any(axis=1).loc[lambda x: x].index
    df = (
        df
        .drop([*character_cols_to_drop, 'length'], axis=1)
        .drop(rows_to_drop, axis=0)
        .set_index('word')
    )

    # Sort with rarest letters first
    df = df[df.sum(axis=0).sort_values(ascending=False).index]

    assert set(df.columns) == characters
    df.to_pickle(WORD_LOOKUP_PATH)

    return df


def find_anagrams(
    text: str,
    all_words: pd.DataFrame = build_word_lookup(),
    stream_results: bool = True,
):
    """
    `text`: the text to be scrambled into anagrams
    `all_words`: a DataFrame with a row for every word in the corpus and
        a column for every character in the corpus. Each value gives the
        number of times a given character occurs in a given word.
    `stream_results`: if True, print anagrams to stdin as they're found

    Main anagram algorithm:
    1. find all words that are contained in `text_letters`
    2. if none found, stop and append any `current_results` to `all_results`
    3. for each word found (if any), branch into a new function call,
        where the letters of the word are stripped from `text_letters`
        and `current_result` contains the word. Go from step 1 again
    """
    # A count of the characters in the input text
    full_text_letters = pd.Series(
        Counter(char for char in text if char in all_words),
        index=all_words.columns
    ).fillna(0).astype(int).values

    all_results = set()

    stack = deque([{
        'text_letters': full_text_letters,
        'words': all_words.index.values,
        'word_letters': all_words.values,
        'current_result': tuple(),
    }])
    # TODO: can I multi-thread this algorithm? Part of it may need to become a function again

    from time import time
    from collections import defaultdict
    times = defaultdict(lambda: [])
    try:
        while stack:
            t0 = time()
            data = stack.pop()
            contained_words = (data['text_letters'] >= data['word_letters']).all(axis=1)
            times['a'].append(time() - t0); t0 = time()

            if not contained_words.any():
                if not data['current_result']:
                    continue
                if any(data['text_letters'] != 0):
                    continue
                if stream_results:
                    print(data['current_result'])
                all_results.add(data['current_result'])
                continue
            times['b'].append(time() - t0); t0 = time()

            # Since we're stripping out letters as we go, we can drop any words
            # that didn't match this time through
            words = data['words'][contained_words]
            word_letters = data['word_letters'][contained_words]
            times['c'].append(time() - t0); t0 = time()

            for i in reversed(range(len(words))):
                t1 = time()
                current_result = tuple([*data['current_result'], words[i]])
                times['AA'].append(time() - t1); t1 = time()

                new_text_letters = data['text_letters'] - word_letters[i]
                times['AB'].append(time() - t1); t1 = time()
                stack.append({
                    'text_letters': new_text_letters,
                    'words': words,
                    'word_letters': word_letters,
                    'current_result': current_result,
                })
                times['BB'].append(time() - t1); t1 = time()

                # TODO: Can I instead build up a mask and put it on the stack each time?
                word_mask = np.ones_like(words).astype(bool)
                word_mask[i] = False
                times['BC'].append(time() - t1); t1 = time()
                words = words[word_mask]
                times['CC'].append(time() - t1); t1 = time()
                word_letters = word_letters[word_mask]
                times['DD'].append(time() - t1); t1 = time()

            times['d'].append(time() - t0); t0 = time()
    except Exception as e:
        print('error:', e)
    finally:
        time_summary = pd.DataFrame({
            k: pd.Series(v).describe() for k, v in times.items()
        })
        time_summary.loc['total'] = time_summary.loc['count'] * time_summary.loc['mean']
        print(time_summary)
    return all_results


if __name__ == '__main__':
    text = ' '.join(sys.argv[1:])

    if text in ('help', '--help', '-h'):
        print(__doc__)
        sys.exit(0)

    stream_results = True

    if text:
        results = find_anagrams(text, stream_results=stream_results)
    else:
        results = []
        for line in sys.stdin:
            results.append(find_anagrams(line, stream_results=stream_results))
        results = '\n\n'.join(results)

    if not stream_results:
        print(results)
