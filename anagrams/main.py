"""
# Anagram Finder
Find all anagrams for text in standard input or command line argument

```bash
echo "text to be scrambled" | python anagrams/main.py

# Or

python anagrams/main.py "text to be scrambled"
```
"""

import sys


def find_anagrams(text):
    return 'anagrams of: ' + text


if __name__ == '__main__':
    text = ' '.join(sys.argv[1:])

    if text in ('help', '--help', '-h'):
        print(__doc__)
        sys.exit(0)

    if text:
        print(find_anagrams(text))
    else:
        for line in sys.stdin:
            print(find_anagrams(line))
