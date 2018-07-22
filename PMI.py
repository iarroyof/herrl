from sklearn.datasets import fetch_20newsgroups
import collections
import math
import windows as w
import numpy as np
import random

def get_closest(k, seq):
    j = 0
    min = None
    for i in range(len(seq)):
        if (seq[i] == 0):
            continue
        x = seq[i] - k
        if (min == None or x < min):
            min = x
            j = i
    return j

def reward(path):
    sum = 0
    for w, k in path:
        sum += k
    return sum

def grad(z, mu):
    g = 1
    return - (2 * g * (z - mu)) / math.log(2)

categories = ['sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories = categories, remove=('headers', 'footers', 'quotes'))

text = newsgroups_train.data[0]

with open('utils/stop_words.txt') as f:
    stop_words = [word for line in f for word in line.split()]

words = text.split()
counter = collections.Counter(words)

# Number of samples of which the agent is going to train of
samples = 4
# Number of tests to check the agent's performance
tests = 4
training_set = []
test_set = []
x = y = None

# Randomly create a training set and a test set that does not include stop words (nouns only)
for i in range(samples + tests):
    while True:
        x = random.choice(words)
        if (x not in stop_words):
            break
    while True:
        y = random.choice(words)
        if (y not in stop_words and y != x):
            break
    if (i < samples):
        training_set.append((x,y))
    else:
        test_set.append((x,y))

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
        x = 0
        for window in windows:
            if word_1 in window and word_2 in window and window.index(word_1) != window.index(word_2):
                x = x + 1
        p_ab = float(x) / len(windows)
        pmi = 0 if (p_ab == 0) else math.log(p_ab / (pa * pb))
        ady_matrix[i,j] = pmi

max_iter = 10
z_k = 0
alpha = 0.1
scale = 1
theta = ady_matrix.mean()
R = 0
R_u = 30

while (R <= R_u):

    # Training the agent
    for sample in training_set:
        iter = 0
        source = sample[0]
        target = sample[1]

        # The word we're currently on
        word_i = source

        # While traveling this path saves the words and pmi
        path = []

        # This path only saves the words that are obtained while traveling
        path_w = []

        while (iter < max_iter) and (target not in path_w):
            z_k = np.random.normal(theta, scale) #mean, standart deviation

            # Save the index of the word we are currently on
            temp_index = words.index(word_i)

            line = list(ady_matrix[temp_index])

            # Removes all ceros from the line so that non-adyacent words are not added in the path
            #line = [value for value in line if value != 0]

            # Searchs for the word closest to the value of z_k, that is adyacent to word_i
            # and that is not already on the path
            while True:
                closest_index = get_closest(z_k, line)
                if (words[closest_index] not in path_w):
                    break

                # In case it was on the path it is deleted so that it isn't considered again
                del line[closest_index]

            # Append word_i to the path of words
            path_w.append(word_i)

            # Append word_i and its pmi to the path
            path.append((word_i, ady_matrix[temp_index, closest_index]))

            #We update theta and the reward value
            R = reward(path)
            theta += alpha * R * grad(iw[word_i], theta)

            # Replace the word we were currently on with the closest one that was found
            word_i = words[closest_index]
            iter += 1
