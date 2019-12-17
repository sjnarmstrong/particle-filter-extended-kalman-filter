import numpy as np

class ExtendedKalmanFilter:
    """
    @param self: The Kalman Filter
    @param f: Mapping from xk_1 to xk
    @param h: Mapping from xk to zk
    @param Jf: Jacobian of f [[df1/dx1, df1/dx2, ..., df1/dxn],
                             [df2/dx1, df2/dx2, ..., df2/dxn],
                             ...,
                             [dfn/dx1, dfn/dx2, ..., dfn/dxn]]
    @param Jh: Jacobian of f [[dh1/dx1, dh1/dx2, ..., dh1/dxn],
                             [dh2/dx1, dh2/dx2, ..., dh2/dxn],
                             ...,
                             [dhn/dx1, dhn/dx2, ..., dhn/dxn]]
    @param r: Value of diagonal cov of v (noise from xk_1 to xk)
    @param q: Value of diagonal cov of w (noise from xk to zk)
    @param n: Size of xk
    @param m: Size of zk
    """
    def __init__(self, f, h, Jf, Jh, r, q, n, m):
        self.f = f    
        self.h = h    
        self.Jf = Jf  
        self.Jh = Jh    
        self.R = r*np.eye(m)    
        self.Q = q*np.eye(n)
        self.I = np.eye(n)

    def predict(self, xk_1pa, Pk_1, k=0):
        """
        Performs the predictions step of the algorithm
        :param xk_1pa: (n) - Previous best estimate of x after the correction step.
        :param Pk_1: (n x n) - Predicted variance of x(k-1)
        :param k: Real number - Used to indicate the current time step. This is given to the function as an optional
        additional parameter.
        :return: Returns the predicted x and its covariance for the current time-step
        """
        x_kpf = self.f(xk_1pa, k)
        Jf = self.Jf(xk_1pa, k)
        P_kpf = np.dot(Jf, np.dot(Pk_1,Jf.T)) + self.Q
        return x_kpf, P_kpf

    def correction(self, x_kpf, P_kpf, zk, k=0):
        """
        Performs the correction step after the algorithm receives a measurement.
        :param x_kpf: (n) - Current prediction of x.
        :param P_kpf: (n x n) - Predicted variance of x.
        :param zk: (m) - Observed measurement.
        :param k: Real number - Used to indicate the current time step. This is given to the function as an optional
        additional parameter.
        :return: Returns the predicted x, its covariance and Kk for the current time-step
        """
        Jh = self.Jh(x_kpf, k)
        PJHT = np.dot(P_kpf, Jh.T)
        invM = np.linalg.inv(np.dot(Jh, PJHT) + self.R)
        Kk = np.dot(PJHT, invM)
        xkpa = x_kpf + np.dot(Kk, zk - self.h(x_kpf, k))
        Pk = np.dot(self.I - np.dot(Kk, Jh), P_kpf)
        return xkpa, Pk, Kk


class IteratedExtendedKalmanFilter(ExtendedKalmanFilter):
    def __init__(self, f, h, Jf, Jh, r, q, n, m, thresh = 1e-6):
        """

        :param f:
        :param h:
        :param Jf:
        :param Jh:
        :param r:
        :param q:
        :param n:
        :param m:
        :param thresh:
        """
        ExtendedKalmanFilter.__init__(self, f, h, Jf, Jh, r, q, n, m)
        self.thresh = thresh

    def correction(self, x_kpf, P_kpf, zk, k=0):
        prevx = x_kpf
        xkpa, Pk, Kk = ExtendedKalmanFilter.correction(self, x_kpf, P_kpf, zk, k)
        diff = prevx - xkpa
        while (diff * diff).sum() > self.thresh:
            prevx = xkpa.copy()
            xkpa, Pk, Kk = ExtendedKalmanFilter.correction(self, xkpa, P_kpf, zk, k)
            diff = prevx - xkpa
        return xkpa, Pk, Kk
