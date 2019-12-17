from Problems import TestProblem3
import numpy as np
from Utils import evaluateAllFiltersLessMemory, convertToLatexBoxAndWhiscar
import os
from multiprocessing import Pool

K=0.2
initP =np.array([[0.5**2,        0,      0,        0],
                 [0     , 0.005**2,      0,        0],
                 [0     ,        0, 0.3**2,        0],
                 [0     ,        0,      0, 0.01**2]]) 


numberOfParticals = 4000
initX = [0.0, 0.0, 0.4, -0.05]

q = 0.001**2
Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
R = np.array([[0.005**2]])

problem = TestProblem3(R,Q)

X0Real = [-0.05, 0.001, 0.7, -0.055]


numSamples = 24

runs = 1000
def evaluateAllFiltersPartial(x):
    return evaluateAllFiltersLessMemory(problem, numSamples, X0Real, numberOfParticals, K,
                               R, Q, initP, initX, [0,1,2,3])

if __name__ == '__main__':
    
    _pool = Pool(8)
    errorsK, errorsP = zip(*_pool.map(evaluateAllFiltersPartial,range(runs)))
            
    errorsK, errorsP = np.array(errorsK), np.array(errorsP)

    convertToLatexBoxAndWhiscar(errorsK[:,0], os.path.join("Outputs","Experement3"), "EKF.tex", "EKF")
    convertToLatexBoxAndWhiscar(errorsK[:,1], os.path.join("Outputs","Experement3"), "IEKF.tex", "IEKF")
    
    convertToLatexBoxAndWhiscar(errorsP[:,0], os.path.join("Outputs","Experement3"), "SIS.tex", "SIS")
    convertToLatexBoxAndWhiscar(errorsP[:,1], os.path.join("Outputs","Experement3"), "SIS_with_jitter.tex", "SIS with jitter")
    convertToLatexBoxAndWhiscar(errorsP[:,2], os.path.join("Outputs","Experement3"), "GPF.tex", "GPF")
    convertToLatexBoxAndWhiscar(errorsP[:,3], os.path.join("Outputs","Experement3"), "GPF_with_jitter..tex", "GPF with jitter")
    convertToLatexBoxAndWhiscar(errorsP[:,4], os.path.join("Outputs","Experement3"), "SIR.tex", "SIR")
    convertToLatexBoxAndWhiscar(errorsP[:,5], os.path.join("Outputs","Experement3"), "SIR_with_jitter.tex", "SIR with jitter")
    #convertToLatexBoxAndWhiscar(errorsP[:,6], os.path.join("Outputs","Experement3"), "SIR_alt_order.tex", "SIR alt order")
    convertToLatexBoxAndWhiscar(errorsP[:,6], os.path.join("Outputs","Experement3"), "SIR_with_times_prev.tex", "SIR times prev")