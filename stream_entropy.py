# This is an implementation of the streaming entropy computation method proposed in

# Lall, A., Sekar, V., Ogihara, M., Xu, J., & Zhang, H. (2006, June). Data streaming algorithms for estimating entropy of network traffic. In ACM SIGMETRICS Performance Evaluation Review (Vol. 34, No. 1, pp. 145-156). ACM.

# This is very seemed to that of:

# Chakrabarti, A., Do Ba, K., & Muthukrishnan, S. (2006). Estimating entropy and entropy norm on data streams. Internet Mathematics, 3(1), 63-78.
# Chakrabarti, A., Cormode, G., & McGregor, A. (2007, January). A near-optimal algorithm for computing the entropy of a stream. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms (pp. 328-335). Society for Industrial and Applied Mathematics.
# Bhuvanagiri, L., & Ganguly, S. (2006, September). Estimating entropy over data streams. In European Symposium on Algorithms (pp. 148-159). Springer, Berlin, Heidelberg.
# Harvey, N. J., Nelson, J., & Onak, K. (2008, October). Sketching and streaming entropy via approximation theory. In Foundations of Computer Science, 2008. FOCS'08. IEEE 49th Annual IEEE Symposium on (pp. 489-498). IEEE.
# Zhao, H. C., Lall, A., Ogihara, M., Spatscheck, O., Wang, J., & Xu, J. (2007, October). A data streaming algorithm for estimating entropies of OD flows. In Proceedings of the 7th ACM SIGCOMM conference on Internet measurement (pp. 279-290). ACM.
# Bhuvanagiri, L., Ganguly, S., Kesh, D., & Saha, C. (2006, January). Simpler algorithm for estimating frequency moments of data streams. In Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm (pp. 708-713). Society for Industrial and Applied Mathematics.
# Ganguly, S., & Cormode, G. (2007). On estimating frequency moments of data streams. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques (pp. 479-493). Springer, Berlin, Heidelberg.
# Li, P. (2009, January). Compressed counting. In Proceedings of the twentieth annual ACM-SIAM symposium on Discrete algorithms (pp. 412-421). Society for Industrial and Applied Mathematics.

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from math import log
import matplotlib.pyplot as plt
from pdb import set_trace as st


class windowStreamer(object):
    def __init__(self, input_file, vectorizer, wsize=10):
        self.file_name = input_file
        self.analyzer = vectorizer.build_analyzer()
        self.tokenizer = vectorizer.build_tokenizer()
        self.wsize = wsize

    def __iter__(self):
        for line in open(self.file_name, mode = "r", encoding = 'latin-1', errors = 'replace'):
            ln = self.tokenizer(line.lower())
            try:
                for i, _ in enumerate(ln):
                    try:
                        #word = ln[i + self.wsize]
                        word = ln[i]
                    except KeyError:
                        continue

                    start = min(len(ln[0:i]), self.wsize)
                    w = ln[i - start:i] + ln[i + 1:i + (self.wsize + 1)]
                    s = " ".join(w)
                    #wi = [word] + self.tokenizer(" ".join(self.analyzer(s)))
                    wi = self.tokenizer(" ".join(self.analyzer(s)))
                    #bow = dictionary.doc2bow(wi)
                    if len(wi) < 2:
                        #stderr.write("%s\n" % wi)
                        continue

                    yield word, wi

            except IndexError:
                break


vectorizer = CountVectorizer(analyzer='word')

mi = False

# For entropy computing
window_size = 10  # The number of words yielded as a window
sampling_stream_size = 20  # The number of windows yielded as stream
input_file = "../mini_dbpedia.txt"

# For mutual information computing
buffer_length = 10

# Window streamming from text document
stream = windowStreamer(input_file, vectorizer, window_size)

if mi:
    buffer = []
    c = 0
    for window in stream:
        if c <= buffer_length:
            buffer.append(window)
            c += 1
        else:
            c = 0
else:
    sampling_stream_buffer = []
    word_buffer = []
    c = 0
    H_window = []
    H_stream = []
    H_symbol = []
    sequence = []
    #latest_buffer = []
    for word, window in stream:
        if c <= sampling_stream_size:  # and latest_buffer == []:
            sampling_stream_buffer.append(window)  #" ".join(window))
            word_buffer.append(word)
            c += 1

        else:
            M = Counter(sum(sampling_stream_buffer, []))
            m = Counter(word_buffer)
            for word in word_buffer:  # Symbol entropies
                try:
                    m_in_M = float(m[word])/float(M[word])
                    M_in_m = float(M[word])/float(m[word])
                    H_window.append(m_in_M * log(M_in_m, 2))
                except ZeroDivisionError:
                    H_window.append(0.0)
            # window entropies
            H_stream.append(sum(H_window))
            H_symbol += H_window
            H_window = []
            sampling_stream_buffer = []
            c = 0
            sequence += word_buffer
            word_buffer = []



plt.plot(H_symbol)
plt.xticks(range(len(sequence)), sequence, rotation='vertical')
plt.grid(True)
plt.tight_layout()
plt.ylabel('The streaming entropy')
plt.show()
