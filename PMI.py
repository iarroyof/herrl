import collections
import math
import windows as w
import numpy as np

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

text = 'When the defendant and his lawyer walked into the court, some of the victim supporters turned their backs on him';

words = text.split()
counter = collections.Counter(words)

pw = {x : float(counter[x])/len(words) for x in counter} # probabilities of each word

iw = {x : pw[x] * math.log(pw[x]) for x in counter} #information on each word

windows = w.get_windows(words, 4)

n = len(words)
ady_matrix = np.zeros(shape=(n,n))
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

path = []
p_words = []
index = 6
z_k = 0
alpha = 0.1
scale = 1
theta = ady_matrix.mean()
R = 0
R_u = 30
iter = 0
word_f = 'defendant'

while (iter < 10) and (R <= R_u) and (word_f not in p_words):
    z_k = np.random.normal(theta, scale) #mean, standart deviation
    word_i = words[index]
    temp_i = index
    line = list(ady_matrix[index])
    while True:
        index = get_closest(z_k, line)
        if (words[index] not in p_words):
            break
        del line[index]

    path.append((words[index], ady_matrix[temp_i,index]))
    p_words.append(words[index])
    R = reward(path)
    theta += alpha * R * grad(iw[word_i], theta)
    iter += 1

print('Path = ' + str(path))
print(theta)
