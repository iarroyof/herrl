from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize as tokenize
from operator import itemgetter
import string
import collections

from windows import get_windows
from Layer import Layer

## Get text
with open('utils/newsgroup_568.txt', 'r') as f:
    text = f.read().replace('\n', ' ')

with open('utils/stop_words.txt') as f:
    stop_words = tokenize(f.read())

words = tokenize(text)
counter = collections.Counter(words)

windows = get_windows(words, 3)

samples = [('result', 'is on', 'time'),
           ('hydrogen', 'is', 'liquid'),
           ('multistaged', 'is on', 'rockets'),
           ('vehicle', 'is', 'white'),
           ('approach', 'is to', 'better'),
           ('phase', 'is on', 'third'),
           ('approach', 'is', 'little'),
           ('sky', 'is in', 'noonday'),
           ('concept', 'is to', 'hardware'),
           ('ammunition', 'is', 'fancy'),
           ('altitudes', 'is at', 'higher'),
           ('checkout', 'is', 'automated'),
           ('consideration', 'is', 'key'),
           ('landing', 'is for', 'soft')]

for a, b, c in samples:
    h = Layer(a)
    h.load_context_windows(windows)
    h.load_set()
    h.set_entropies()
    h.set_total_entropy()
    h.set_words_before_and_after()

    j = Layer(c)
    j.load_context_windows(windows)
    j.load_set()
    j.set_entropies()
    j.set_total_entropy()
    j.set_words_before_and_after()


    if (h.windows == []):
        continue

    h_min = min(h.windows_after_xs, key=itemgetter(0))[0]

    j_min = min(j.windows_before_xs, key=itemgetter(0))[0]

    hs = [(e, w) for e, w in h.windows_after_xs if (e == h_min)]
    js = [(e, w) for e, w in j.windows_before_xs if (e == j_min)]

    relations = []
    r = []

    print(a + ' ' + b + ' ' + c)

    for e_h, w_h in hs:
        for e_j, w_j in js:
            if (not set(w_h).isdisjoint(w_j)):
                if (w_h[-1] == w_j[0]):
                    relations.append( ((e_h, w_h), (w_j, e_j)) )
                    r.append(w_h + w_j[1:])

    for w in relations:
        print(w)
    for w in r:
        print(w)

    print('#' * 20)
