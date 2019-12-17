import numpy as np
f = lambda x, k : x
h = f
X = np.array([[0],[2],[3],[4]])
Xind = [0,1,2,3]
meanNoiseX = [0]
Q = np.eye(1)*0.0001
R = np.eye(1)*0.0002**2
numberOfSamples = 4
k=0

z = [4]
_6sqrtr = 6*0.2

newXMean = f(X, k)
X2 = newXMean + np.random.multivariate_normal(meanNoiseX,Q,numberOfSamples)
z_est = h(X2, k)

w = multivariate_normal.logpdf(z_est,z,R)*w
b = w.max()
w = np.exp(w - b)
w = (w/np.sum(w))

ind = np.where(diff>_6sqrtr)[0]
ind2 = np.random.choice(Xind, len(ind), True, 1.0-diff/np.sum(diff))
X2[ind] = X2[ind2]
z_est[ind] = z_est[ind2]
#
#self.w = multivariate_normal.logpdf(z_est,z,self.R)*self.w
#b = self.w.max()
#self.w = np.exp(self.w - b)
#self.w = (self.w/np.sum(self.w))
#return self.X[np.argmax(self.w)], self.w
