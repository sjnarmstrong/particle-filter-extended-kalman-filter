import numpy as np

class KalmanFilter:
    def __init__(self, A, B, C, R, Q):
        self.A = A    #control of prev x on x
        self.B = B    #any external control (Accelerator)
        self.C = C    #x to z
        self.R = R    #cov of v
        self.Q = Q    #cov of w 
    def predict(self, xk_1, u, Pk_1):
        xbar = np.dot(self.A,xk_1)+np.dot(self.B,u)
        Pkbar = np.dot(self.A, np.dot(Pk_1, self.A))+self.Q
        return xbar, Pkbar
    def correction(self, Pkbar, xbar, zk):
        Pk_d_CT = np.dot(Pkbar, self.C.T)
        invM = np.linalg.inv(np.dot(self.C, Pk_d_CT) + self.R)
        Kk = np.dot(Pk_d_CT, invM)
        xk = xbar + np.dot(Kk, zk - np.dot(self.C, xbar))
        Pk = np.dot(1 - np.dot(Kk,self.C), Pkbar)
        return xk, Pk, Kk

class KalmanFilter1D:
    def __init__(self, A, B, C, R, Q):
        self.A = A    #control of prev x on x
        self.B = B    #any external control (Accelerator)
        self.C = C    #x to z
        self.R = R    #cov of v
        self.Q = Q    #cov of w 
    def predict(self, xk_1, u, Pk_1):
        xbar = np.dot(self.A,xk_1)+np.dot(self.B,u)
        Pkbar = np.dot(self.A, np.dot(Pk_1, self.A))+self.Q
        return xbar, Pkbar
    def correction(self, Pkbar, xbar, zk):
        Pk_d_CT = np.dot(Pkbar, self.C)
        invM = 1/(np.dot(self.C, Pk_d_CT) + self.R)
        Kk = np.dot(Pk_d_CT, invM)
        xk = xbar + np.dot(Kk, zk - np.dot(self.C, xbar))
        Pk = np.dot(1 - np.dot(Kk,self.C), Pkbar)
        return xk, Pk, Kk
    
    
"""
Test from http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies
"""

kf = KalmanFilter1D(1, 0, 1, 0.1, 0)
xk = 0
pk = 1

readings = [0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45]

print("zk\txk-1\tPk\tbarxk\tbarP\tKK\txk\tPk")
for reading in readings:
    xk_1=xk
    pk_1=pk
    xbar, pbar = kf.predict(xk_1, 0,pk_1)
    xk, pk, Kk = kf.correction(pbar, xbar, reading)
    print("{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}\t{6:.2f}\t{7:.2f}"
          .format(*(reading, xk_1, pk_1, xbar, pbar, Kk, xk, pk)))