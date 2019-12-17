from Problems import TestProblem1
import numpy as np
from Utils import evaluateAllFiltersLessMemory, convertToLatexBoxAndWhiscar
import os
from multiprocessing import Pool

problem = TestProblem1(0.01,0.001)
numSamples = 100
numberOfParticals = 8000
initP =np.eye(4)*0.5

runs = 100
def evaluateAllFiltersPartial(x):
    return evaluateAllFiltersLessMemory(problem, numSamples, [0.0,-1.0,0.0,np.pi], numberOfParticals, 0.4,
                               problem.r*np.eye(problem.m), problem.q*np.eye(problem.n), initP, [0.0, 0.0, 0.0, 0.0], [0,1])

if __name__ == '__main__':
    
    _pool = Pool(8)
    errorsK, errorsP = zip(*_pool.map(evaluateAllFiltersPartial,range(runs)))
            
    errorsK, errorsP = np.array(errorsK), np.array(errorsP)

    convertToLatexBoxAndWhiscar(errorsK[:,0], os.path.join("Outputs","Experement1"), "EKF.tex", "EKF")
    convertToLatexBoxAndWhiscar(errorsK[:,1], os.path.join("Outputs","Experement1"), "IEKF.tex", "IEKF")
    
    convertToLatexBoxAndWhiscar(errorsP[:,0], os.path.join("Outputs","Experement1"), "SIS.tex", "SIS")
    convertToLatexBoxAndWhiscar(errorsP[:,1], os.path.join("Outputs","Experement1"), "SIS_with_jitter.tex", "SIS with jitter")
    convertToLatexBoxAndWhiscar(errorsP[:,2], os.path.join("Outputs","Experement1"), "GPF.tex", "GPF")
    convertToLatexBoxAndWhiscar(errorsP[:,3], os.path.join("Outputs","Experement1"), "GPF_with_jitter..tex", "GPF with jitter")
    convertToLatexBoxAndWhiscar(errorsP[:,4], os.path.join("Outputs","Experement1"), "SIR.tex", "SIR")
    convertToLatexBoxAndWhiscar(errorsP[:,5], os.path.join("Outputs","Experement1"), "SIR_with_jitter.tex", "SIR with jitter")
    #convertToLatexBoxAndWhiscar(errorsP[:,6], os.path.join("Outputs","Experement1"), "SIR_alt_order.tex", "SIR alt order")
    convertToLatexBoxAndWhiscar(errorsP[:,6], os.path.join("Outputs","Experement1"), "SIR_with_times_prev.tex", "SIR times prev")