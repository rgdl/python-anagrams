# Anagram Finder
Find all anagrams for text in standard input or command line argument

# To setup
```sh
poetry env use <PATH-TO-PYTHON-3.8>
poetry install
```

# To use
```sh
poetry run python anagrams/main.py "text to be scrambled"
```

To run linter:
```sh
flake8 .
```

To run tests:
```sh
poetry run pytest tests
```
