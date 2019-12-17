import numpy as np

#class ProblemType: 
#    def generateSamples(self, numSamples, startloc):
#        sampledx = np.empty((numSamples+1,self.n),np.float64)
#        sampledx[0,:] = startloc
#        
#        noisyx = np.empty((numSamples,self.n),np.float64)
#        realz = np.empty((numSamples,self.m),np.float64)
#        noisyz = np.empty((numSamples,self.m),np.float64)
#        
#               
#        if self.r is None:
#            R = np.eye(self.m)
#        elif np.shape(self.r)==():
#            R = self.r*np.eye(self.m)
#        else:
#            R = self.r
#        
#        if self.q is None:
#            Q = np.eye(self.n)
#        elif np.shape(self.q)==():
#            Q = self.q*np.eye(self.n)
#        else:
#            Q = self.q
#            
#            
#        for i, x in enumerate(sampledx[:-1]):
#            realz[i] = self.h(sampledx[i], i+1)
#            
#            noisyx[i] = sampledx[i] + np.random.multivariate_normal(np.zeros(self.n),Q)
#            noisyz[i] = self.h(noisyx[i], i+1) + np.random.multivariate_normal(np.zeros(self.m),R)
#            
#            sampledx[i+1] = self.f(sampledx[i], i+2)
#        return sampledx[:-1], realz, noisyx, noisyz

class ProblemType: 
    
    def generateSamples(self, numSamples, startloc):
        sampledx = np.empty((numSamples+1,self.n),np.float64)
        sampledx[0,:] = startloc
        
        realz = np.empty((numSamples,self.m),np.float64)
        noisyz = np.empty((numSamples,self.m),np.float64)
        
               
        if self.r is None:
            R = np.eye(self.m)
        elif np.shape(self.r)==():
            R = self.r*np.eye(self.m)
        else:
            R = self.r
        
        if self.q is None:
            Q = np.eye(self.n)
        elif np.shape(self.q)==():
            Q = self.q*np.eye(self.n)
        else:
            Q = self.q
            
            
        for i, x in enumerate(sampledx[:-1]):
            realz[i] = self.h(sampledx[i], i+1)
            
            noisyz[i] = realz[i] + np.random.multivariate_normal(np.zeros(self.m),R)
            
            sampledx[i+1] = self.f(sampledx[i], i+2) + np.random.multivariate_normal(np.zeros(self.n),Q)
        return sampledx[:-1], realz, noisyz
    """
    def generateSamples(self, numSamples, startloc):
        sampledx = np.empty((numSamples,self.n),np.float64)
        
        prevx = startloc
        
        realz = np.empty((numSamples,self.m),np.float64)
        noisyz = np.empty((numSamples,self.m),np.float64)
        
               
        if self.r is None:
            R = np.eye(self.m)
        elif np.shape(self.r)==():
            R = self.r*np.eye(self.m)
        else:
            R = self.r
        
        if self.q is None:
            Q = np.eye(self.n)
        elif np.shape(self.q)==():
            Q = self.q*np.eye(self.n)
        else:
            Q = self.q
            
            
        for i, x in enumerate(sampledx[:-1]):
            prevx = self.f(prevx, i+2) + np.random.multivariate_normal(np.zeros(self.n),Q)
            sampledx[i] = prevx
            
            realz[i] = self.h(sampledx[i], i+1)
            
            noisyz[i] = realz[i] + np.random.multivariate_normal(np.zeros(self.m),R)
            
            
        return sampledx, realz, noisyz
    """
class TestProblem1(ProblemType):
    def __init__(self, r, q):
        self.n,self.m = 4,2
        self.r,self.q = r, q
        self.mult = 0
        self.mult4 = 0
    def generateSamples(self, numSamples, startloc = [0.0,-1.0,0.0,np.pi]):
        self.mult = np.pi*4/numSamples
        self.mult4 = self.mult*4
        return ProblemType.generateSamples(self, numSamples, startloc)
    def f(self,x,k):
        return np.array([np.sin(x[2]+self.mult),
                np.cos(x[3]+self.mult4),
                x[2]+self.mult,
                x[3]+self.mult4])
    def h(self, x,k):
        return np.dot(x,[[1,4],[0.8,1],[0,0],[0,0]])
    def bulkf(self,x,k):
        return np.array([np.sin(x[:,2]+self.mult),
                np.cos(x[:,3]+self.mult4),
                x[:,2]+self.mult,
                x[:,3]+self.mult4]).T
    def bulkh(self, x,k):
        return self.h(x,k)
    
    def Jf(self, x,k):
        return np.array([[0 ,0 ,np.cos(x[2]+self.mult)  ,0                            ],
                         [0 ,0 ,0                        ,-np.sin(x[3]+self.mult4)    ],
                         [0 ,0 ,1                        ,0                           ],
                         [0 ,0 ,0                        ,1                           ]])
    
    def Jh(self, x,k):
        return np.array([[1   , 0.8,   0,   0],
                         [4   ,   1,   0,   0]])

    
class TestProblem2(ProblemType):
    def __init__(self, r, q):
        self.n,self.m = 1,1
        self.r,self.q = r, q
        self.k = 0
    def generateSamples(self, numSamples, startloc = [0.0]):
        return ProblemType.generateSamples(self, numSamples, startloc)
            
    def f(self,x,k):
        return (0.5*x) + (25*x/(1+(x*x))) + 8*np.cos(1.2*(k-1))
    def h(self, x,k):
        return (x*x)/20
    def bulkf(self,x,k):
        return self.f(x,k)
    def bulkh(self, x,k):
        return self.h(x,k)
    
    def Jf(self, x,k):
        x2=x*x
        return np.array(0.5+ 25*(x2-1)/((x2+1)**2) )
    
    def Jh(self, x,k):
        return np.array(x/10)
    
    
class TestProblem3(ProblemType):
    def __init__(self, r=0.005**2, q=0.001**2):
        self.n,self.m = 4,1
        self.r = r
        self.q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        self.k = 0
    def generateSamples(self, numSamples, startloc = [-0.05, 0.001, 0.7, -0.055]):
        return ProblemType.generateSamples(self, numSamples, startloc)
            
    def f(self,x,k):
        return np.dot(x,[[1,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,1,1]])
    def h(self, x,k):
        return np.nan_to_num(np.arctan2(x[2],x[0]))
    def bulkf(self,x,k):
        return self.f(x,k)
        return np.dot(x,[[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
    def bulkh(self, x,k):
        return np.nan_to_num(np.arctan2(x[:,2],x[:,0]))
    
    def Jf(self, x,k):
        return np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
    
    def Jh(self, x,k):
        x2py2=np.nan_to_num(1.0/(x[0]**2+x[2]**2))
        return np.array([[-x[2]*x2py2, 0, x[0]*x2py2, 0]])
    
    
    
    
class TestProblem4(ProblemType):
    def __init__(self, r=0.005**2, q=0.001**2):
        self.n,self.m = 4,2
        self.r = r
        self.q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        self.k = 0
    def generateSamples(self, numSamples, startloc = [-0.05, 0.001, 0.7, -0.055]):
        return ProblemType.generateSamples(self, numSamples, startloc)
            
    def f(self,x,k):
        return np.dot(x,[[1,0,0,0],[1,1,0,0],[0,0,1,0],[0,0,1,1]])
    def h(self, x,k):
        return np.array([np.nan_to_num(np.arctan2(x[2],x[0]))
        ,x[2]**2+x[0]**2])
    def bulkf(self,x,k):
        return self.f(x,k)
        return np.dot(x,[[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
    def bulkh(self, x,k):
        return np.array([np.nan_to_num(np.arctan2(x[:,2],x[:,0]))
        ,x[:,2]**2+x[:,0]**2]).T
    
    def Jf(self, x,k):
        return np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
    
    def Jh(self, x,k):
        x2py2=np.nan_to_num(1.0/(x[0]**2+x[2]**2))
        return np.array([[-x[2]*x2py2, 0, x[0]*x2py2, 0],
                         [2*x[0], 0, 2*x[2], 0]])


        