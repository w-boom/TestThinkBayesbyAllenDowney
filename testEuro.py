from empiricaldist import Pmf
import numpy as np
import matplotlib.pyplot as plt

hypos = np.linspace(0, 1, 101) # 对应x， 假设的概率
prior = Pmf(1, hypos) # 先验，假设x为1的均匀分布
prior.normalize()

print(prior)

likelihood_heads = hypos
likelihood_tails = 1 - hypos

likelihood = {
    'H':likelihood_heads,
    'T':likelihood_tails
}

dataset = 'H' * 140 + 'T' * 110

def updata_euro(pmf, dataset):
    for data in dataset:
        pmf *= likelihood[data]
        
    pmf.normalize()
    
posterior = prior.copy()
updata_euro(posterior, dataset)

posterior.plot(label = '140 heads out of 250', color = 'C4')
ax = plt.gca()
ax.set(xlabel = 'Proportion of heads(x)', ylabel = 'Probability', title = 'Posterior distribution of x')

print(posterior.idxmax())
print(posterior(posterior.idxmax()))