import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def truncated_power_law(a, m):
    """https://stackoverflow.com/questions/24579269/
                            sample-a-truncated-integer-power-law-in-python"""

    x = np.arange(1, m + 1, dtype='float')
    pmf = 1 / x ** a
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(1, m + 1), pmf))

a, m = 2, 10
d = truncated_power_law(a=a, m=m)

N = 10 ** 4
sample = d.rvs(size=N)

plt.hist(sample, bins=np.arange(m) + 0.5)
plt.show()