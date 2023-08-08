from empiricaldist import Pmf
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Congress Problem:
    There are 538 members of the United States Congress. 
    Suppose we audit their investment portfolios and find that 
    312 of them out-perform the market. 
    Let's assume that an honest member of Congress has only a 
    50% chance of out-performing the market, but a dishonest 
    member who trades on inside information has a 90% chance. 
    How many members of Congress are honest?
"""

def make_binomial(n, p):
    """Make a binomial distribution.

    n: number of trials
    p: probability of success

    returns: Pmf representing the distribution of k
    """
    ks = np.arange(n+1)
    ps = binom.pmf(ks, n, p)
    return Pmf(ps, ks)

def add_dist(pmf1, pmf2):
    """Compute the distribution of a sum"""
    res = Pmf()
    for q1, p1 in pmf1.items():
        for q2, p2 in pmf2.items():
            q = q1 + q2
            p = p1 + p2
            res[q] = res(q) + p
    return res

n = 538

table = pd.DataFrame(index = range(n+1), columns = range(n+1))

for n_honest in range(0, n+1):
    n_dishonest = n - n_honest
        
    dist_honest = make_binomial(n_honest, 0.5)
    dist_dishonest = make_binomial(n_dishonest, 0.9)
    dist_total = Pmf.add_dist(dist_honest, dist_dishonest)    

    table[n_honest] = dist_total
    
data = 312
likelihood = table.loc[312]

hypos = np.arange(n+1)
prior = Pmf(1, hypos)

posterior = prior * likelihood 
posterior.normalize()

print('the number of honest members is:', posterior.mean())
posterior.plot(label='posterior')
ax = plt.gca()
ax.legend()
ax.set(xlabel='Number of honest members of Congress',
         ylabel='PMF')
    