from empiricaldist import Pmf
import numpy as np
import matplotlib.pyplot as plt

class Cookie(object):
    
    def __init__(self, data, kinds):
        self.bowlsNum = data.shape[0]
        self.kindsNum = data.shape[1]
        self.kinds = kinds # 种类
        hypos = np.arange(bowlsNum)
        
        # 先验，P（碗），每个碗的概率是相同的
        self.prior = Pmf(1, hypos)
        self.prior.normalize()
   
        # 似然P（香草|碗）
        self.likelihood = data
        self.posterior = self.prior
        
        
    def Updata(self, timeNum, kind):
        index = self.kinds.index(kind)
        for time in range(timeNum):
            self.posterior *= self.likelihood[..., index]
            self.posterior.normalize()
            
        return self.posterior
    

    
if __name__ == '__main__':
    bowlsNum = 101
    kinds = ['vanilla', 'chocolate']
    kindsNum = len(kinds)
    
    # data包含每种饼干种类的比例
    data = np.zeros((bowlsNum, kindsNum), dtype = float)
    data[..., 0] = np.arange(0.00, 1.01, 0.01)
    data[..., 1] = 1.00 - data[..., 0]
    
    pp = Cookie(data, kinds)

    like1 = pp.Updata(10, 'vanilla')
    like1.plot(label='vanilla', color='C5')
    
    like2 = pp.Updata(3, 'chocolate')
    like2.plot(label='chocolate', color='C4')
    
    ax = plt.gca()
    ax.legend()
    ax.set(xlabel = 'Bowl #', ylabel = 'PMF', title = 'Posterior')

    # 概率最大的碗
    maxBowl = like2.idxmax()
    print(maxBowl)
    
            
            
        