from Problems import TestProblem3
import numpy as np
from Utils import evaluatePFFilterLessMemory, convertToLatexBoxAndWhiscar
import os
from multiprocessing import Pool
from functools import partial

K=0.2


initP =np.array([[0.5**2,        0,      0,        0],
                 [0     , 0.005**2,      0,        0],
                 [0     ,        0, 0.3**2,        0],
                 [0     ,        0,      0, 0.01**2]]) 


initX = [0.0, 0.0, 0.4, -0.05]

q = 0.001**2
Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
R = np.array([[0.005**2]])

problem = TestProblem3(R,Q)

X0Real = np.array([-0.05, 0.001, 0.7, -0.055])
weightings = [-15.0,8.0,15.0,-8.0]

numSamples = 24

runs = 20
def evaluateAllFiltersPartial(numberOfParticals, currX):
    return evaluatePFFilterLessMemory(problem, numSamples, X0Real, numberOfParticals, K,
                               R, Q, initP, initX, [0,1,2,3])

if __name__ == '__main__':
    
    _pool = Pool(8)
    
    errorMeans = []
    errorUQs = []
    errorLQs = []
    errorMedians = []
    errorMins = []
    errorMaxs = []
    numparticalsArray = range(200,10200,200)
    for numberOfParticals in numparticalsArray:
        loopPfunc = partial(evaluateAllFiltersPartial, numberOfParticals)
        errorsP, = zip(*_pool.map(loopPfunc,range(runs)))
        errorsP = np.array(errorsP)
        errorMeans.append(np.mean(errorsP, axis=0))
        errorUQs.append(np.percentile(errorsP, 75, axis=0))
        errorLQs.append(np.percentile(errorsP, 25, axis=0))
        errorMedians.append(np.percentile(errorsP, 50, axis=0))
        errorMins.append(np.min(errorsP, axis=0))
        errorMaxs.append(np.max(errorsP, axis=0))
        
    print(np.array(errorMedians))
    
    if not os.path.exists(os.path.join("Outputs","Experement6")):
        os.makedirs(os.path.join("Outputs","Experement6"))
    
    np.savez(os.path.join("Outputs","Experement6","outputData.npz"), 
             np.array(errorMeans), 
             np.array(errorUQs), 
             np.array(errorLQs), 
             np.array(errorMedians), 
             np.array(errorMins), 
             np.array(errorMaxs), 
             np.array(numparticalsArray))