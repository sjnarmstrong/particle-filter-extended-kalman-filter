import numpy as np
from scipy.stats import multivariate_normal

def scaleLogW(logw):
    b = logw.max()
    scale = b+np.log(np.sum(np.exp(logw - b)))
    return logw - scale


class ParticleFilter:
    #@profile
    def __init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz):
        self.f = f
        self.h = h
        self.R = R   
        self.Q = Q
        self.X = initX
        self.Xind = np.arange(numberOfSamples)
        self.numberOfSamples = numberOfSamples
        self.meanNoiseX = np.repeat(0,initX.shape[-1])
        self.logw = np.log(np.ones(numberOfSamples)/numberOfSamples)

    #@profile
    def iterate(self, z, k=0):
        newXMean = self.f(self.X, k)
        self.X = newXMean + np.random.multivariate_normal(self.meanNoiseX,self.Q,self.numberOfSamples)
        z_est = self.h(self.X, k)
        self.logw = multivariate_normal.logpdf(z_est,z,self.R)+self.logw
        self.logw = scaleLogW(self.logw)
        return self.X[np.argmax(self.logw)], self.logw
    


class ParticleFilterWithJitter:
    #@profile
    def __init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz, K):
        self.f = f
        self.h = h
        self.R = R   
        self.Q = Q
        self.X = initX
        self.Xind = np.arange(numberOfSamples)
        self.numberOfSamples = numberOfSamples
        self.meanNoiseX = np.repeat(0,initX.shape[-1])
        self.logw = np.log(np.ones(numberOfSamples)/numberOfSamples)
        d = initX.shape[-1]
        self.Jk = np.eye(d)*K*(numberOfSamples**(-d))

    #@profile
    def iterate(self, z, k=0):
        newXMean = self.f(self.X, k)
        
        E = np.ptp(self.X,axis=0)
        self.X = newXMean + np.random.multivariate_normal(self.meanNoiseX,self.Q,self.numberOfSamples)
        self.X += np.random.multivariate_normal(self.meanNoiseX,(self.Jk*E),self.numberOfSamples)
        
        z_est = self.h(self.X, k)
        self.logw = multivariate_normal.logpdf(z_est,z,self.R)+self.logw
        self.logw = scaleLogW(self.logw)
        return self.X[np.argmax(self.logw)], self.logw


class ParticleFilterSIR(ParticleFilter):
    #@profile
    def __init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz):
        ParticleFilter.__init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz)
    #@profile
    def iterate(self, z, k=0):
        self.X=self.X[np.random.choice(self.Xind, self.numberOfSamples, True, np.exp(self.logw))]
        newXMean = self.f(self.X, k)
        self.X = newXMean + np.random.multivariate_normal(self.meanNoiseX,self.Q,self.numberOfSamples)
        z_est = self.h(self.X, k)
        self.logw = multivariate_normal.logpdf(z_est,z,self.R)
        self.logw = scaleLogW(self.logw)
        
        
        return self.X[np.argmax(self.logw)], self.logw
        #return np.dot(self.w, self.X)/np.sum(self.w), self.w

"""
class ParticleFilterSIR2(ParticleFilter):
    #@profile
    def __init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz):
        ParticleFilter.__init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz)
    #@profile
    def iterate(self, z, k=0):
        newXMean = self.f(self.X, k)
        self.X = newXMean + np.random.multivariate_normal(self.meanNoiseX,self.Q,self.numberOfSamples)
        z_est = self.h(self.X, k)
        self.logw = multivariate_normal.logpdf(z_est,z,self.R)
        self.logw = scaleLogW(self.logw)
        
        self.X=self.X[np.random.choice(self.Xind, self.numberOfSamples, True, np.exp(self.logw))]
        
        return self.X[np.argmax(self.logw)], self.logw
        #return np.dot(self.w, self.X)/np.sum(self.w), self.w
"""

class ParticalFilterGeneric(ParticleFilter):
    #@profile
    def __init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz, Nt=None):
        ParticleFilter.__init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz)
        self.Nt = Nt if Nt is not None else 0.6*numberOfSamples
    #@profile
    def iterate(self, z, k=0):
        outx, _ = ParticleFilter.iterate(self, z, k)
        outx = outx.copy()
        if 1.0/np.linalg.norm(np.exp(self.logw))**2 <= self.Nt:
            ind = np.random.choice(self.Xind, self.numberOfSamples, True, np.exp(self.logw))
            self.X = self.X[ind]
            self.logw = self.logw[ind]
            self.logw = scaleLogW(self.logw)
        
        return outx, self.logw
    

class ParticalFilterGenericWithJitter(ParticleFilterWithJitter):
    #@profile
    def __init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz, K, Nt=None):
        ParticleFilterWithJitter.__init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz, K)
        self.Nt = Nt if Nt is not None else 0.6*numberOfSamples
    #@profile
    def iterate(self, z, k=0):
        outx, _ = ParticleFilterWithJitter.iterate(self, z, k)
        
        if 1.0/np.linalg.norm(np.exp(self.logw))**2 <= self.Nt:
            ind = np.random.choice(self.Xind, self.numberOfSamples, True, np.exp(self.logw))
            self.X = self.X[ind]
            self.logw = self.logw[ind]
            self.logw = scaleLogW(self.logw)
        
        return outx, self.logw

class ParticleFilterWithWeightMomentum(ParticleFilter):
    #@profile
    def __init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz):
        ParticleFilter.__init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz)

    #@profile
    def iterate(self, z, k=0):
        newXMean = self.f(self.X[np.random.choice(self.Xind, self.numberOfSamples, True, np.exp(self.logw))], k)
        self.X = newXMean + np.random.multivariate_normal(self.meanNoiseX,self.Q,self.numberOfSamples)
        z_est = self.h(self.X, k)
        self.logw = multivariate_normal.logpdf(z_est,z,self.R)+self.logw
        self.logw = scaleLogW(self.logw)
        return self.X[np.argmax(np.exp(self.logw))], np.exp(self.logw)
    
class ParticleFilterSIRWithJitter(ParticleFilter):
    #@profile
    def __init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz, K):
        ParticleFilter.__init__(self, f, h, R, Q, initX, numberOfSamples, sizeOfz)
        d = initX.shape[-1]
        self.Jk = np.eye(d)*K*(numberOfSamples**(-d))
    #@profile
    def iterate(self, z, k=0):
        newXMean = self.f(self.X[np.random.choice(self.Xind, self.numberOfSamples, True, np.exp(self.logw))], k)
        E = np.ptp(self.X,axis=0)
        self.X = newXMean + np.random.multivariate_normal(self.meanNoiseX,self.Q,self.numberOfSamples)
        self.X += np.random.multivariate_normal(self.meanNoiseX,(self.Jk*E),self.numberOfSamples)
        z_est = self.h(self.X, k)
        self.logw = multivariate_normal.logpdf(z_est,z,self.R)
        self.logw = scaleLogW(self.logw)
        return self.X[np.argmax(self.logw)], self.logw
    
    
    """
    def iterate(self, z, k=0):
        newXMean = self.f(self.X[np.random.choice(self.Xind, self.numberOfSamples, True, self.w)], k)
        E = np.ptp(self.X,axis=0)
        self.X = newXMean + np.random.multivariate_normal(self.meanNoiseX,self.Q,self.numberOfSamples)
        self.X += np.random.multivariate_normal(self.meanNoiseX,(self.Jk*E),self.numberOfSamples)
        z_est = self.h(self.X, k)
        self.w = multivariate_normal.logpdf(z_est,z,self.R)
        b = self.w.max()
        self.w = np.exp(self.w - b)
        self.w = (self.w/np.sum(self.w))
        return self.X[np.argmax(self.w)], self.w
        #return np.dot(self.w, self.X)/np.sum(self.w), self.w
        
    def iterate(self, z, k=0):
        newXMean = self.f(self.X, k)
        self.X = newXMean + np.random.multivariate_normal(self.meanNoiseX,self.Q,self.numberOfSamples)
        z_est = self.h(self.X, k)
        logw = multivariate_normal.logpdf(z_est,z,self.R)
        logw = scaleLogW(logw)
        
        xout = self.X[np.argmax(logw)]
        
        E = np.ptp(self.X,axis=0)
        ind = np.random.choice(self.Xind, self.numberOfSamples, True, np.exp(logw))
        self.X=self.X[ind] + np.random.multivariate_normal(self.meanNoiseX,(self.Jk*E),self.numberOfSamples)
        self.logw = logw[ind]
        self.logw = scaleLogW(self.logw)
        return xout, self.logw"""

