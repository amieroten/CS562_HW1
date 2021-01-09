''' A short script that takes as an argument a filename
    pointing to a file containing one sentence per line,
    split on " " into tokens. The script then performs
    a number of count-based analyses, and reports back
    the results. '''

import sys
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pickle as pkl
from pathlib import Path
from math import isclose

from matplotlib import pyplot as plt
from nltk import pos_tag
from nltk.corpus import stopwords
import numpy as np

# This is really slow. :(
def get_ngrams(corpus: List[str], n: int,
               include_stops: bool = True) -> Counter:
    # Also remove possessive suffix.
    stops = [word.upper() for word in stopwords.words("english")] + ["'S"]
    ngrams = Counter()
    for sentence in corpus:
        sentence = sentence.strip("\n")
        # Remove words that are empty strings.
        words = [word for word in sentence.split(" ") if word != ""]
        # Remove stopwords, if selected.
        if not include_stops:
            words = [word for word in words if word not in stops]
        for index in range(len(words)-(n-1)):
            gram = tuple([words[i] for i in range(index, index+n)])
            ngrams[gram] += 1

    return ngrams


def ngram_probabilities(corpus: List[str], n: int,
                        n_min_onegrams: Optional[Counter] = None,
                        ngrams: Optional[Counter] = None,
                        include_stops: bool = False) -> Dict[Tuple, float]:
    probabilities = {}
    if n_min_onegrams is None and n != 1:
        n_min_onegrams = get_ngrams(corpus, 1, include_stops)
    if ngrams is None:
        ngrams = get_ngrams(corpus, n, include_stops)

    if n == 1:
        word_total = sum(ngrams.values())
        for key, value in ngrams.items():
            probabilities[key] = float(value) / word_total
        assert isclose(1.0, sum(probabilities.values()))
    else:
        for key, value in ngrams.items():
            history = key[0:-1]
            probabilities[key] = float(value) / n_min_onegrams[history]
    return probabilities


def calculate_PMI(unigram_prob: Dict, bigram_prob: Dict,
                  bigrams: Counter, threshold: int = 0) -> Dict:
    PMI = {}
    for key, value in bigram_prob.items():
        if bigrams[key] >= threshold:
            PMI[key] = value / (unigram_prob[(key[0],)]*unigram_prob[(key[1],)])
    return PMI


if __name__ == "__main__":
    include_stops = True
    corpus_path = sys.argv[1]
    with open(corpus_path, "r") as f:
        corpus = f.readlines()

    # Serialize unigrams and bigrams since this
    # takes a long time to process.
    if include_stops:
        unigrams_path = Path("unigrams_stops.pkl")
        bigrams_path = Path("bigrams_stops.pkl")
    else:
        unigrams_path = Path("unigrams.pkl")
        bigrams_path = Path("bigrams.pkl")

    if unigrams_path.exists():
        with open(unigrams_path, "rb") as f:
            unigrams = pkl.load(f)
    else:
        unigrams = get_ngrams(corpus, 1, include_stops=include_stops)
        with open(unigrams_path, 'wb') as f:
            pkl.dump(unigrams, f)

    if bigrams_path.exists():
        with open(bigrams_path, "rb") as f:
            bigrams = pkl.load(f)
    else:
        bigrams = get_ngrams(corpus, 2, include_stops=include_stops)
        with open(bigrams_path, 'wb') as f:
            pkl.dump(bigrams, f)

    types = len(unigrams.keys())
    print("Q1: There are", types, "types in this corpus.")
    tokens = sum(unigrams.values())
    print("Q2: There are", tokens, "unigram tokens in this corpus.")

    # Rank-frequency plot based on plots from https://en.wikipedia.org/wiki/Zipf's_law
    ranked = unigrams.most_common()
    log_frequency = [np.log(item[1]) for item in ranked]
    log_rank = [np.log(i) for i in range(1, len(log_frequency)+1)]
    words = [item[0][0] for item in ranked]
    plt.plot(log_rank, log_frequency)
    plt.xlabel("Log Rank")
    plt.ylabel("Log Frequency")
    if include_stops:
        plt.title("Rank-Frequency Plot for Central News Agency section\nof Gigawords Corpus")
        plt.savefig("rank-freq.png")
    else:
        plt.title("Rank-Frequency Plot for Central News Agency section\nof Gigawords Corpus (no stops)")
        plt.savefig("rank-freq_no_stops.png")
    plt.close()

    print("Q4: The thirty most common words are: ", words[:30])

    unigram_probs = ngram_probabilities(corpus, n = 1, n_min_onegrams=None, ngrams=unigrams)
    bigram_probs = ngram_probabilities(corpus, n = 2, n_min_onegrams=unigrams, ngrams=bigrams)

    pmi = calculate_PMI(unigram_probs, bigram_probs, bigrams, threshold=100)
    pmi_ordered = sorted(list(pmi.items()), key=lambda x: x[1], reverse=True)
    top_30 = pmi_ordered[:30]
    top_10 = pmi_ordered[:10]

    # Get corresponding bigram frequencies:
    for pair in top_30:
        print("Bigram:", pair[0], ", \n\tPMI:",  pair[1],
              ",\n\tbigram frequency:", bigrams[pair[0]],
              ",\n\t", pair[0][0], " unigram frequency:",
              unigrams[(pair[0][0],)], ", \n\t", pair[0][1],
              " unigram frequency:", unigrams[(pair[0][1],)])

    print("PMI for New York:", pmi[("NEW","YORK")])
    print("\tBigram frequency:",bigrams[("NEW","YORK")])
    print("\tNEW unigram frequency:",unigrams[("NEW",)])
    print("\tYORK unigram frequency:", unigrams[("YORK",)])
