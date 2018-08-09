from sklearn.datasets import fetch_20newsgroups
import collections
import math
import windows as w
import numpy as np
import random
from nltk.tokenize import word_tokenize as tokenize
import string
from pdb import set_trace as st


def get_closest(k, seq):
    dif = {i: seq[i] - k for i in seq}
    j = min(dif, key=dif.get)

    return j, abs(abs(k) - abs(seq[j]))


def reward(path, op="mean"):
    if len(path) < 3: op = "mean"
    if op == "sum":
        return sum([k for w, k in path])
    elif op == "mean":
        return np.mean([k for w, k in path])
    elif op == "median":
        return np.median([k for w, k in path])
    elif op == "var":
        return np.var([k for w, k in path])


def penalty(x, m=1.0, b=0.0):
    return m * x + b


def grad(x, mean, sigma):
    # More info at: https://www.wolframalpha.com/input/?i=p(x)+%3D+(1%2F%E2%88%9A(2*%CF%80*s%5E2))*e%5E-((1%2F(2*s%5E2))+*(x-m)%5E2)
    k = x -  mean
    z = sigma ** 2
    f = (1/(2*z)) * (k ** 2)
    return - (k * np.exp(f)) / (math.sqrt(2 * math.pi) * (k ** (3/2)))



categories = ['sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories = categories, remove=('headers', 'footers', 'quotes'))

table = str.maketrans({key: None for key in string.punctuation})
text = newsgroups_train.data[0].lower().translate(table)

with open('utils/stop_words.txt') as f:
    stop_words = [word for line in f for word in line.split()]

words = tokenize(text)
counter = collections.Counter(words)

# Number of samples of which the agent is going to train of
samples = 10
# Number of tests to check the agent's performance
tests = 4
training_set = []
test_set = []
x = y = None

 # Randomly create a training set and a test set that does not include stop words (nouns only)
 for i in range(samples + tests):
     while True:
         x = random.choice(words)
         # Since the strings are unicode, they need to be encoded so that we can compare them
         x = x.encode('ascii','ignore')
         if (x not in stop_words):
             break
     while True:
         y = random.choice(words)
         y = y.encode('ascii','ignore')
         if (y not in stop_words and y != x):
             break
     if (i < samples):
         training_set.append((x,y))

# Probabilities of each word
pw = {x : float(counter[x])/len(words) for x in counter}

# Information value on each word
iw = {x : pw[x] * math.log(pw[x]) for x in counter}

# The windows of which the pmi values are going to be obtained
windows = w.get_windows(words, 4)

n = len(words)
ady_matrix = np.zeros(shape=(n,n))

# Obtains the values of the adyacent matrix
for i in range(len(words)):
    word_1 = words[i]
    pa = pw[word_1]
    for j in range(len(words)):
        word_2 = words[j]
        pb = pw[word_2]
        for window in windows:
            if word_1 in window and word_2 in window and window.index(word_1) != window.index(word_2):
                ady_matrix[i,j] = 1
                break

max_iter = 50
z_k = 0
alpha = 0.01
scale = 0.5
mean_mi = ady_matrix.mean()
theta = mean_mi
print("Initial model's theta: {}".format(theta))
R = 0
R_u = 1.5
tilt = 1.0
r_func = "var"

#st()

while (R <= R_u):

    # Training the agent
    for sample in training_set:
        print("Current sample pair: {}".format(sample))
        iter = 0
        source, target = sample

        # The word we're currently on
        word_i = source

        # While traveling this path saves the words and pmi
        path = []

        # This path only saves the words that are obtained while traveling
        path_w = []

        while (iter < max_iter): # and (target not in path_w):
            z_k = np.random.normal(theta, scale) #mean, standart deviation

            # Save the index of the word we are currently on
            temp_index = words.index(word_i)

            line = list(ady_matrix[temp_index])
            # Removes all ceros from the line so that non-adyacent words are not added in the path
            line = {i: x for i, x in enumerate(line) if x > 0.0}
            #st()
            # Searchs for the word closest to the value of z_k, that is adyacent to word_i
            # and that is not already on the path
            while not len(line) == 1:
                closest_index, delta = get_closest(z_k, line)
                if (words[closest_index] not in path_w):
                    break
                # In case it was on the path it is deleted so that it isn't considered again
                del line[closest_index]

            # Append word_i to the path of words
            path_w.append(word_i)

            # Append word_i and its pmi to the path
            path.append((word_i, ady_matrix[temp_index, closest_index]))
            #st()
            #We update theta and the reward value
            R = reward(path, r_func) #- penalty(delta, tilt)
            if R < -1000.0: continue
            theta = theta + alpha * R * grad(iw[word_i], theta, scale)
            print("New reward: {}".format(R))
            print("New theta: {}".format(theta))
            # Replace the word we were currently on with the closest one that was found
            word_i = words[closest_index]
            iter += 1
            print("Current path: {}".format(path))
