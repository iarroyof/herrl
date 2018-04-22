# This is an implementation of the streaming entropy computation method proposed in

# [1] Lall, A., Sekar, V., Ogihara, M., Xu, J., & Zhang, H. (2006, June). Data streaming algorithms for estimating entropy of network traffic. In ACM SIGMETRICS Performance Evaluation Review (Vol. 34, No. 1, pp. 145-156). ACM.

# This is very seemed to that of:

# [2] Chakrabarti, A., Do Ba, K., & Muthukrishnan, S. (2006). Estimating entropy and entropy norm on data streams. Internet Mathematics, 3(1), 63-78.
# [3] Chakrabarti, A., Cormode, G., & McGregor, A. (2007, January). A near-optimal algorithm for computing the entropy of a stream. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms (pp. 328-335). Society for Industrial and Applied Mathematics.
# [4] Bhuvanagiri, L., & Ganguly, S. (2006, September). Estimating entropy over data streams. In European Symposium on Algorithms (pp. 148-159). Springer, Berlin, Heidelberg.
# [5] Harvey, N. J., Nelson, J., & Onak, K. (2008, October). Sketching and streaming entropy via approximation theory. In Foundations of Computer Science, 2008. FOCS'08. IEEE 49th Annual IEEE Symposium on (pp. 489-498). IEEE.
# [6] Zhao, H. C., Lall, A., Ogihara, M., Spatscheck, O., Wang, J., & Xu, J. (2007, October). A data streaming algorithm for estimating entropies of OD flows. In Proceedings of the 7th ACM SIGCOMM conference on Internet measurement (pp. 279-290). ACM.
# [7] Bhuvanagiri, L., Ganguly, S., Kesh, D., & Saha, C. (2006, January). Simpler algorithm for estimating frequency moments of data streams. In Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm (pp. 708-713). Society for Industrial and Applied Mathematics.
# [8] Ganguly, S., & Cormode, G. (2007). On estimating frequency moments of data streams. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques (pp. 479-493). Springer, Berlin, Heidelberg.
# [9] Li, P. (2009, January). Compressed counting. In Proceedings of the twentieth annual ACM-SIAM symposium on Discrete algorithms (pp. 412-421). Society for Industrial and Applied Mathematics.

# Furthermore, the implementation for entropy was also modified to incorporate Mutual Information between each word and the stream it is sampled from.

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from math import log
import matplotlib.pyplot as plt
import argparse


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


parser = argparse.ArgumentParser(description='Computation of entropy and mutual information of each item in a stream of words.')
parser.add_argument('--wsize', type=int, default=10, help='An integer inidicating window size within the input stream. (default=10)')
parser.add_argument('--ssize', type=int, default=20, help='An integer indicating the batch size yielded as input stream. (default=20)')
parser.add_argument('--MI', action='store_true', help='Activates Mutual Information as information theoretic measure.')
parser.add_argument('--input', required=True, help='The input file for streaming.')

args = parser.parse_args()

mi = args.MI

# Stream entropy computing parameters
window_size = args.wsize  # The number of words yielded as a window
sampling_stream_size = args.ssize  # The number of windows yielded as stream
word_char = 'word'  # Only words supported for now..
input_file = args.input  # "../mini_dbpedia.txt"

# Window streamming from text document
vectorizer = CountVectorizer(analyzer=word_char)
stream = windowStreamer(input_file, vectorizer, window_size)

if mi:
    sampling_stream_buffer = []
    word_buffer = []
    c = 0
    H_window = []  # The entropy of each center word of a window computed w.r.t the the set of windows associated to all them.
    H_stream = []  # 
    H_symbol = []  # All the resulting H_window s concatenated
    sequence = []

    for word, window in stream:
        if c <= sampling_stream_size:
            sampling_stream_buffer.append(window)
            word_buffer.append(word)
            c += 1

        else:
            M = Counter(sum(sampling_stream_buffer, []))
            m = Counter(word_buffer)
            for word in word_buffer:  # Symbol entropies
                try:
                    m_in_M = float(m[word])/float(M[word])
                    M_in_m = float(M[word])/float(m[word])
                    H_window.append(m_in_M * log(M_in_m, 2))  # Inverted ratio as in [2], Eq. 1.
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

else:
    sampling_stream_buffer = []
    word_buffer = []
    c = 0
    H_window = []  # The entropy of each center word of a window computed w.r.t the the set of windows associated to all them.
    H_stream = []  # 
    H_symbol = []  # All the resulting H_window s concatenated
    sequence = []
    #latest_buffer = []
    for word, window in stream:
        if c <= sampling_stream_size:  # and latest_buffer == []:
            sampling_stream_buffer.append(window)  #" ".join(window))
            word_buffer.append(word)
            c += 1

        else:
            #M = Counter(sum(sampling_stream_buffer, []))
            M = sampling_stream_size * window_size
            m = Counter(word_buffer)
            for word in word_buffer:  # Symbol entropies
                try:
                    #m_in_M = float(m[word])/float(M[word])
                    m_in_M = float(m[word])/float(M)
                    #M_in_m = float(M[word])/float(m[word])
                    M_in_m = float(M)/float(m[word])
                    H_window.append(m_in_M * log(M_in_m, 2))  # Inverted ratio as in [2], Eq. 1.
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
