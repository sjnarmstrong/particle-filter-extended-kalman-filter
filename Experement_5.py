from Problems import TestProblem3
import numpy as np
from Utils import evaluateImportantAllFiltersLessMemory, convertToLatexBoxAndWhiscar
import os
from multiprocessing import Pool
from functools import partial

K=0.2



numberOfParticals = 4000

q = 0.001**2
Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
R = np.array([[0.005**2]])

problem = TestProblem3(R,Q)

X0Real = np.array([-0.05, 0.001, 0.7, -0.055])
weightings = [-15.0,8.0,15.0,-8.0]

numSamples = 24

runs = 1000
def evaluateAllFiltersPartial(initX, currX):
    initP = np.eye(4)*(1e-6+(X0Real-initX)**2)
    return evaluateImportantAllFiltersLessMemory(problem, numSamples, X0Real, numberOfParticals, K,
                               R, Q, initP, initX, [0,1,2,3])

if __name__ == '__main__':
    
    _pool = Pool(8)
    
    for axis in range(4):
        errorMeans = []
        errorUQs = []
        errorLQs = []
        errorMedians = []
        #errorMins = []
        #errorMaxs = []
        X0RealValues = []
        for percentageOfActual in np.linspace(0,1,100):
            X0RealValue = X0Real.copy()
            X0RealValue[axis] += weightings[axis]*percentageOfActual
            X0RealValues.append(X0RealValue)
            loopPfunc = partial(evaluateAllFiltersPartial, X0RealValue)
            errorsK, errorsP = zip(*_pool.map(loopPfunc,range(runs)))
                    
            errorlist = np.append(errorsK,errorsP,axis=1)
            errorMeans.append(np.mean(errorlist, axis=0))
            errorUQs.append(np.percentile(errorlist, 75, axis=0))
            errorLQs.append(np.percentile(errorlist, 25, axis=0))
            errorMedians.append(np.percentile(errorlist, 50, axis=0))
            #errorMins.append(np.min(errorlist, axis=0))
            #errorMaxs.append(np.max(errorlist, axis=0))
            
        print(np.array(errorMedians)[:,0])
        print(np.array(errorMedians)[:,1])
        
        if not os.path.exists(os.path.join("Outputs","Experement5")):
            os.makedirs(os.path.join("Outputs","Experement5"))
        
        np.savez(os.path.join("Outputs","Experement5","outputDatax_"+str(axis)+".npz"), 
                 np.array(errorMeans), 
                 np.array(errorUQs), 
                 np.array(errorLQs), 
                 np.array(errorMedians), 
                 #np.array(errorMins), 
                 #np.array(errorMaxs), 
                 np.array(X0RealValues))