from math import log

class Layer(object):
    single = False

    xs = None
    xf = None
    windows = None
    set_words = None
    entropies = None
    entropy = None

    windows_before_xs = None
    windows_after_xs = None

    def __init__(self, xs, xf = None):
        self.entropy = 0
        self.xs = xs
        self.xf = xf if xf is not None else xs
        self.single = False if xf is not None else True
        self.windows = []
        self.entropies = []
        self.set_words = set()

        self.windows_before_xs = []
        self.windows_after_xs = []

    def load_set(self):
        dummy_set = set()
        for window in self.windows:
            dummy_set = dummy_set.union(set(window))

        dummy_set -= set([self.xs, self.xf])
        self.set_words = dummy_set

    def get_set(self):
        return self.set_words

    def load_context_windows(self, windows):
        for window in windows:
            if (self.xs in window and self.xf in window):
                self.windows.append(window)

    def get_candidates(self, stop_words):
        while True:
            if (len(self.set_words) == 0):
                return None, None

            x, y = random.sample(self.set_words, 2)
            if (x == y or x == self.xs or x == self.xf or y == self.xs or y == self.xf or x in stop_words or y in stop_words):
                continue
            return x, y

    def set_entropies(self):
        for i in range(len(self.windows)):
            self.entropies.append((self.set_entropy(self.windows[i]), self.windows[i]))

    def set_entropy(self, window_j):
        n = len(self.set_words)
        m = len(set(window_j).intersection(self.set_words))
        return log(n/m)

    def set_total_entropy(self):
        N = len(self.set_words)
        self.entropy = - log(1/N) if N is not 0 else 0

    def set_words_before_and_after(self):
        if not self.single:
            return
        for h, window in self.entropies:
            index = window.index(self.xs)
            w_1 = window[:index]
            if (len(w_1) > 0):
                self.windows_before_xs.append((h, w_1))
            w_2 = window[index + 1:]
            if (len(w_2) > 0):
                self.windows_after_xs.append((h, w_2))
