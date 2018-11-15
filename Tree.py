from anytree import AnyNode, LevelOrderIter
import random

class Tree(object):
    root = None
    windows = None
    stop_words = None
    max_childs = 0
    max_depth = 0

    def __init__(self, max_childs, max_depth, windows, stop_words = []):
        self.max_childs = max_childs
        self.max_depth = max_depth
        self.windows = windows
        self.stop_words = stop_words

    def set_root_with_samples(self, samples):
        samples_copy = samples.copy()

        while (True):
            if (len(samples_copy) == 0):
                print('Samples where exahusted')
                return

            a, b = random.choice(samples_copy)

            l = Layer(a, b)
            l.load_context_windows(self.windows)

            if (len(l.windows) == 0):
                samples_copy.remove((a,b))
            else:
                l.load_set()
                l.set_entropies()
                l.set_total_entropy()
                break

        self.root = AnyNode(id = str(a) + ' ' + str(b), layer = l, depth = 0)

    def set_root(self, a, b):
        l = Layer(a, b)
        l.load_context_windows(self.windows)
        l.load_set()
        l.set_entropies()
        l.set_total_entropy()
        self.root = AnyNode(id = str(a) + ' ' + str(b), layer = l, depth = 0)

    def set_child(self, parent, word_1, word_2, depth = 0):
        layer_both = Layer(word_1, word_2)
        layer_word_1 = Layer(word_1)
        layer_word_2 = Layer(word_2)

        layer_both.load_context_windows(self.windows)
        layer_both.load_set()
        layer_both.set_entropies()
        layer_both.set_total_entropy()
        layer_word_1.load_context_windows(self.windows)
        layer_word_1.load_set()
        layer_word_1.set_entropies()
        layer_word_1.set_total_entropy()
        layer_word_2.load_context_windows(self.windows)
        layer_word_2.load_set()
        layer_word_2.set_entropies()
        layer_word_2.set_total_entropy()

        child_both = AnyNode(id = str(layer_both.xs) + ' ' + str(layer_both.xf), layer = layer_both, parent = parent, depth = depth)
        child_word_1 = AnyNode(id = str(layer_word_1.xs), layer = layer_word_1, parent = parent, depth = depth)
        child_word_2 = AnyNode(id = str(layer_word_2.xs), layer = layer_word_2, parent = parent, depth = depth)

        if (depth < self.max_depth):
            for i in range(self.max_childs):
                new_word_1, new_word_2 = layer_both.get_candidates(self.stop_words)
                self.set_child(child_both, new_word_1, new_word_2, depth + 1)

                new_word_1, new_word_2 = layer_word_1.get_candidates(self.stop_words)
                self.set_child(child_word_1, new_word_1, new_word_2, depth + 1)

                new_word_1, new_word_2 = layer_word_2.get_candidates(self.stop_words)
                self.set_child(child_word_2, new_word_1, new_word_2, depth + 1)

    def construct(self):
        if (self.root is None):
            print('missing root')
            return

        for i in range(self.max_childs):
            word_1, word_2 = self.root.layer.get_candidates(self.stop_words)
            self.set_child(self.root, word_1, word_2, depth = 1)
