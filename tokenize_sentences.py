''' A short script that takes as an argument a filename
    pointing to a file containing sentences which need
    to be tokenized into sentences and words. Individual
    sentences will be placed on each line, split up into
    words using nltk's word tokenizer. These lines are then
    printed to stdout. '''

import sys
import string

from nltk import tokenize

if __name__ == "__main__":
    all_punct = set(list(string.punctuation) + ['""', "--", "``", "''"])

    if len(sys.argv) < 2:
        sys.stderr.write("Must provide file to be tokenized!")
        exit()

    with open(sys.argv[1], "r") as f:
        text = f.read()
        text = text.replace("\n", " ")

    # Process assisted by example code in
    # NLTK tokenize documentation.
    sentences = tokenize.sent_tokenize(text, language='english')
    sentences_tokenized = [tokenize.word_tokenize(sentence) for sentence in sentences]
    sentences_tokenized_nopunc = [" ".join([token.upper() for token in sentence if token not in all_punct]) for sentence in sentences_tokenized]

    for sentence in sentences_tokenized_nopunc:
        print(sentence)

