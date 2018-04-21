from math import log
from collections import Counter
import random
import matplotlib.pyplot as plt
from pdb import set_trace as st

def MI(sequences,i,j):
    try:
        Pi = Counter(sequence[i] for sequence in sequences)
        Pj = Counter(sequence[j] for sequence in sequences)
        Pij = Counter((sequence[i],sequence[j]) for sequence in sequences)
    except IndexError:
        return None

    return sum(float(Pij[(x,y)])*log(float(Pij[(x,y)])/float(Pi[x]*Pj[y])) if Pi[x]*Pj[y] != 0 else 0.0 for x,y in Pij)


low = 1
high = 10

f = open("mini_dbpedia.txt")
A = [  # you'll need to pad the end of your strings so that they're all the
       # same length for this to play nice with numpy
    "MTSKLG--SLKP",
    "MAASLA-ASLPE",
    "MTSKLGAASLPE"]

q="MSLAASLKGPTE"

position = [(random.randint(low, high), random.randint(low, high)) 
                                                        for k in range(10)]
A = map(str.lower, f.readlines())[:50]

MIs = []
for a, b in position:
    for index in range(len(A)):
        S = A[:index] + A[index+1 :]
        print(a, b, index)
        #mi = MI([s for s in map(list, S)], a, b)
        mi = MI([s for s in map(str.split, S)], a, b)
        MIs.append(mi)

plt.plot(MIs)
plt.ylabel('MI by line')
plt.show()
