from Problems import TestProblem1,TestProblem2,TestProblem3,TestProblem4
from ParticalFilter import ParticleFilter, ParticleFilterWithJitter, ParticalFilterGeneric, ParticalFilterGenericWithJitter, ParticleFilterSIR, ParticleFilterSIR2
from ParticalFilter import ParticleFilterWithWeightMomentum, ParticleFilterSIRWithJitter
import matplotlib.pyplot as plt
import numpy as np
from ExtendedKalmanFilter import ExtendedKalmanFilter, IteratedExtendedKalmanFilter
from Utils import evalueateFilters, do_batch_evaluation
import os
from scipy.stats import multivariate_normal


if __name__ == '__main__':
    
    TestsToDo=[False, True, False, False, False, False, False]
    
    if TestsToDo[6]:
        numberOfParticals = 4000
        initP =np.array([[0.5**2,        0,      0,        0],
                         [0     , 0.005**2,      0,        0],
                         [0     ,        0, 0.3**2,        0],
                         [0     ,        0,      0, 0.01**2]]) 
        
        q = 0.001**2
        Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        R = np.array([[0.005**2]])
        
        problem = TestProblem3(R,Q)
        initX = [0.0, 0.0, 0.4, -0.05]
        #initX = [-0.05, 0.001, 0.7, -0.055]
        
        initXpf = np.random.multivariate_normal(initX, initP, numberOfParticals)
        
        
        #0.0001**2*np.eye(problem.n)
        
        
        runs = 100
        filteredXKalman, allPs, filteredXP, allParticals, errorsK, errorsP = do_batch_evaluation(
                problem = problem,
                initialProbX= [-0.05, 0.001, 0.7, -0.055],
                R=R, Q=Q, initXpf=initXpf,
                numberOfParticals=numberOfParticals,
                initialXoutKalman = initX,
                initialPK = initP,
                numSamples = 24,
                numberOfRuns = 20)
        print(errorsK)
        print("Minimum K error: ", np.min(errorsK, axis =0))
        print("LQ K error: ", np.percentile(errorsK, 25, axis=0))
        print("Med K error: ", np.percentile(errorsK, 50, axis=0))
        print("UQ K error: ", np.percentile(errorsK, 75, axis=0))
        print("Maximum K error: ", np.max(errorsK, axis =0))
        print("Mean K error: ", np.mean(errorsK, axis =0))
        print("________________________________________________________")
        print("Minimum PF error: ", np.min(errorsP, axis =0))
        print("LQ PF error: ", np.percentile(errorsP, 25, axis=0))
        print("Med PF error: ", np.percentile(errorsP, 50, axis=0))
        print("UQ PF error: ", np.percentile(errorsP, 75, axis=0))
        print("Maximum PF error: ", np.max(errorsP, axis =0))
        print("Mean PF error: ", np.mean(errorsP, axis =0))
    if TestsToDo[0]:
        print("""
        Test 1: Test partical filters and kalman filters with sin cos problem
        """)
        problem = TestProblem1(0.01,0.01)
        numSamples = 50
        numberOfParticals = 1000
        initP =np.eye(4)*0.5
        initX = np.random.multivariate_normal([0.0, 0.0, 0, 0], initP, numberOfParticals)
        
        pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                            R = 0.1*np.eye(problem.m), Q = 0.01*np.eye(problem.n),
                            initX = initX, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = 0.1*np.eye(problem.m), Q = 0.01*np.eye(problem.n),
                            initX = initX, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.01)
        
        pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                            R = 0.1*np.eye(problem.m), Q = 0.01*np.eye(problem.n),
                            initX = initX, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = 0.1*np.eye(problem.m), Q = 0.01*np.eye(problem.n),
                            initX = initX, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.01)
        
        pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                            R = 0.1*np.eye(problem.m), Q = 0.01*np.eye(problem.n),
                            initX = initX, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = 0.1*np.eye(problem.m), Q = 0.01*np.eye(problem.n),
                            initX = initX, numberOfSamples = numberOfParticals, sizeOfz = problem.m,
                            K = 0.01)
        
        pf32= ParticleFilterSIR2(f = problem.bulkf, h = problem.bulkh, 
                            R = 0.1*np.eye(problem.m), Q = 0.01*np.eye(problem.n),
                            initX = initX, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                            R = 0.1*np.eye(problem.m), Q = 0.01*np.eye(problem.n),
                            initX = initX, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        
        ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                   ,0.1*np.eye(problem.m),0.01*np.eye(problem.n),problem.n,problem.m)
        iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                   ,0.1*np.eye(problem.m),0.01*np.eye(problem.n),problem.n,problem.m)
        
        sampledx, realz, noisyz = problem.generateSamples(numSamples = numSamples)
        
        filteredXKalman, allPs, filteredXP, allParticals = evalueateFilters(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf32, pf4], np.array([0.0,0.0,0.0,0.0]), initP, noisyz)
        
        print("Input")
        plt.plot(range(numSamples),sampledx[:,0],sampledx[:,1])
        plt.show()
        print("PF")
        plt.plot(range(numSamples),sampledx[:,0],sampledx[:,1])
        plt.plot(range(numSamples),filteredXP[0,:,0],filteredXP[0,:,1])
        plt.plot(range(numSamples),filteredXP[1,:,0],filteredXP[1,:,1])
        plt.show()
        print("GPF")
        plt.plot(range(numSamples),sampledx[:,0],sampledx[:,1])
        plt.plot(range(numSamples),filteredXP[2,:,0],filteredXP[2,:,1])
        plt.plot(range(numSamples),filteredXP[3,:,0],filteredXP[3,:,1])
        plt.show()
        print("SIR")
        plt.plot(range(numSamples),sampledx[:,0],sampledx[:,1])
        plt.plot(range(numSamples),filteredXP[4,:,0],filteredXP[4,:,1])
        plt.show()
        print("Momentum")
        plt.plot(range(numSamples),sampledx[:,0],sampledx[:,1])
        plt.plot(range(numSamples),filteredXP[5,:,0],filteredXP[5,:,1])
        plt.show()
        print("SIRJitter")
        plt.plot(range(numSamples),sampledx[:,0],sampledx[:,1])
        plt.plot(range(numSamples),filteredXP[6,:,0],filteredXP[6,:,1])
        plt.show()
        
        
        plt.plot(range(numSamples),filteredXP[2,:,0],filteredXP[2,:,1])
        plt.plot(range(numSamples),filteredXP[4,:,0],filteredXP[4,:,1])
        plt.show()
        
        plt.plot(range(numSamples),filteredXKalman[0,:,0],filteredXKalman[0,:,1])
        plt.show()
        plt.plot(range(numSamples),filteredXKalman[1,:,0],filteredXKalman[1,:,1])
        plt.show()
    
    if TestsToDo[1]:
        print(
        """
        Test 2: Get errors of the problems with the parametners given in the referance
        """)
        
        numberOfParticals = 4000
        initP =np.array([[0.5**2,        0,      0,        0],
                         [0     , 0.005**2,      0,        0],
                         [0     ,        0, 0.3**2,        0],
                         [0     ,        0,      0, 0.01**2]]) 
        
        q = 0.001**2
        Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        R = np.array([[0.005**2]])
        
        problem = TestProblem3(R,Q)
        initX = [0.0, 0.0, 0.4, -0.05]
        #initX = [-0.05, 0.001, 0.7, -0.055]
        
        initXpf = np.random.multivariate_normal(initX, initP, numberOfParticals)
        
        
        #0.0001**2*np.eye(problem.n)
        
        
        runs = 100
        errorsK = np.empty((runs, 2),dtype = np.float64)
        errorsP = np.empty((runs, 8),dtype = np.float64)
        for run in range(runs):
        
            pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf32 = ParticleFilterSIR2(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)        
            
            
            ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                       ,R,Q,problem.n,problem.m)
            iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                       ,R,Q,problem.n,problem.m)
            
            sampledx, realz, noisyz = problem.generateSamples(numSamples = 24)
            #sampledx, realz, noisyz = problem.generateSamples(numSamples = 1000)
            
            
            filteredXKalman, allPs, filteredXP, allParticals = evalueateFilters(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf32, pf4], np.array(initX), initP.copy(), noisyz)
            
            errorsK[run] = np.sqrt(((filteredXKalman - sampledx)**2).mean(axis=(1,2)))
            errorsP[run] = np.sqrt(((filteredXP - sampledx)**2).mean(axis=(1,2)))
            if run%10 == 0: 
                print(run)
        
        print("Minimum K error: ", np.min(errorsK, axis =0))
        print("LQ K error: ", np.percentile(errorsK, 25, axis=0))
        print("Med K error: ", np.percentile(errorsK, 50, axis=0))
        print("UQ K error: ", np.percentile(errorsK, 75, axis=0))
        print("Maximum K error: ", np.max(errorsK, axis =0))
        print("Mean K error: ", np.mean(errorsK, axis =0))
        print("________________________________________________________")
        print("Minimum PF error: ", np.min(errorsP, axis =0))
        print("LQ PF error: ", np.percentile(errorsP, 25, axis=0))
        print("Med PF error: ", np.percentile(errorsP, 50, axis=0))
        print("UQ PF error: ", np.percentile(errorsP, 75, axis=0))
        print("Maximum PF error: ", np.max(errorsP, axis =0))
        print("Mean PF error: ", np.mean(errorsP, axis =0))
    
    if TestsToDo[2]:
        print(
        """
        Test 3: Get errors of baring only tracking for randomly produced scenarios
        """)
    
    
        numberOfParticals = 4000
        initP = 0.01*np.array([[0.5**2,        0,      0,        0],
                         [0     , 0.005**2,      0,        0],
                         [0     ,        0, 0.3**2,        0],
                         [0     ,        0,      0, 0.01**2]]) 
        initP2 = np.array([[1,        0,      0,        0],
                         [0     , 0.5**2,      0,        0],
                         [0     ,        0, 1,        0],
                         [0     ,        0,      0, 0.5**2]]) 
        
        q = 0.1**2
        Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        R = np.array([[0.005**2]])
        
        problem = TestProblem3(R,Q)
        
        
        runs = 100
        errorsK = np.empty((runs, 2),dtype = np.float64)
        errorsP = np.empty((runs, 8),dtype = np.float64)
        for run in range(runs):
            
            
            initXRN = np.random.multivariate_normal([10,0,10,0], initP2)
            initX = initXRN + np.random.multivariate_normal([0,0,0,0], initP)
            initXpf = np.random.multivariate_normal(initX, initP, numberOfParticals)
        
        
            pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            pf5 = ParticalFilterAsDescribed(q, R, initXpf, numberOfParticals)
            
            
            
            ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                       ,R,Q,problem.n,problem.m)
            iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                       ,R,Q,problem.n,problem.m)
            
            sampledx, realz, noisyz = problem.generateSamples(numSamples = 24, startloc=initXRN)
            
            
            filteredXKalman, allPs, filteredXP, allParticals = evalueateFilters(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf4, pf5], initX, initP, noisyz)
            
            errorsK[run] = ((filteredXKalman - sampledx)**2).mean(axis=(1,2))
            errorsP[run] = ((filteredXP - sampledx)**2).mean(axis=(1,2))
            if run%10 == 0: 
                print(run)
        
        print("Minimum K error: ", np.min(errorsK, axis =0))
        print("LQ K error: ", np.percentile(errorsK, 25, axis=0))
        print("Med K error: ", np.percentile(errorsK, 50, axis=0))
        print("UQ K error: ", np.percentile(errorsK, 75, axis=0))
        print("Maximum K error: ", np.max(errorsK, axis =0))
        print("Mean K error: ", np.mean(errorsK, axis =0))
        print("________________________________________________________")
        print("Minimum PF error: ", np.min(errorsP, axis =0))
        print("LQ PF error: ", np.percentile(errorsP, 25, axis=0))
        print("Med PF error: ", np.percentile(errorsP, 50, axis=0))
        print("UQ PF error: ", np.percentile(errorsP, 75, axis=0))
        print("Maximum PF error: ", np.max(errorsP, axis =0))
        print("Mean PF error: ", np.mean(errorsP, axis =0))
    
    
    if TestsToDo[3]:
        print(
        """
        Test 4: Get results of baring and squared distrance tracking problem for randdom scenarios
        """)
        numberOfParticals = 4000
        initP = 0.01*np.array([[0.5**2,        0,      0,        0],
                         [0     , 0.005**2,      0,        0],
                         [0     ,        0, 0.3**2,        0],
                         [0     ,        0,      0, 0.01**2]]) 
        initP2 = np.array([[1,        0,      0,        0],
                         [0     , 0.5**2,      0,        0],
                         [0     ,        0, 1,        0],
                         [0     ,        0,      0, 0.5**2]]) 
        
        q = 0.1**2
        Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        R = np.array([[0.005**2,0],[0,0.005]])
        
        problem = TestProblem4(R,Q)
        
        
        runs = 100
        errorsK = np.empty((runs, 2),dtype = np.float64)
        errorsP = np.empty((runs, 7),dtype = np.float64)
        for run in range(runs):
            
            
            initXRN = np.random.multivariate_normal([10,0,10,0], initP2)
            initX = initXRN + np.random.multivariate_normal([0,0,0,0], initP)
            initXpf = np.random.multivariate_normal(initX, initP, numberOfParticals)
        
        
            pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
            
            
            
            ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                       ,R,Q,problem.n,problem.m)
            iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                       ,R,Q,problem.n,problem.m)
            
            sampledx, realz, noisyz = problem.generateSamples(numSamples = 24, startloc=initXRN)
            
            
            filteredXKalman, allPs, filteredXP, allParticals = evalueateFilters(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf4], initX, initP, noisyz)
            
            errorsK[run] = ((filteredXKalman - sampledx)**2).mean(axis=(1,2))
            errorsP[run] = ((filteredXP - sampledx)**2).mean(axis=(1,2))
            if run%10 == 0: 
                print(run)
        
        print("Minimum K error: ", np.min(errorsK, axis =0))
        print("LQ K error: ", np.percentile(errorsK, 25, axis=0))
        print("Med K error: ", np.percentile(errorsK, 50, axis=0))
        print("UQ K error: ", np.percentile(errorsK, 75, axis=0))
        print("Maximum K error: ", np.max(errorsK, axis =0))
        print("Mean K error: ", np.mean(errorsK, axis =0))
        print("________________________________________________________")
        print("Minimum PF error: ", np.min(errorsP, axis =0))
        print("LQ PF error: ", np.percentile(errorsP, 25, axis=0))
        print("Med PF error: ", np.percentile(errorsP, 50, axis=0))
        print("UQ PF error: ", np.percentile(errorsP, 75, axis=0))
        print("Maximum PF error: ", np.max(errorsP, axis =0))
        print("Mean PF error: ", np.mean(errorsP, axis =0))
    
    
    if TestsToDo[4]:
        print(
        """
        Test 5: Plot Graphs for a random problem using parings only information and the problem from the ref
        """)
        
        numberOfParticals = 4000
        initP =np.array([[0.5**2,        0,      0,        0],
                         [0     , 0.005**2,      0,        0],
                         [0     ,        0, 0.3**2,        0],
                         [0     ,        0,      0, 0.01**2]]) 
        
        q = 0.001**2
        Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        R = np.array([[0.005**2]])
        
        problem = TestProblem3(R,Q)
        initX = [0.0, 0.0, 0.4, -0.05]
        #initX = [-0.05, 0.001, 0.7, -0.055]
        
        initXpf = np.random.multivariate_normal(initX, initP, numberOfParticals)
        
        
        #0.0001**2*np.eye(problem.n)
        
        
        pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
        
        pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
        
        pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
        
        pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        pf5 = ParticalFilterAsDescribed(q, R, initXpf, numberOfParticals)
        
        
        
        ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                   ,R,Q,problem.n,problem.m)
        iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                   ,R,Q,problem.n,problem.m)
        
        sampledx, realz, noisyz = problem.generateSamples(numSamples = 24)
        
        
        filteredXKalman, allPs, filteredXP, allParticals = evalueateFilters(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf4, pf5], initX, initP, noisyz)
       
     
        ylimmin = min(np.min(sampledx[:,2]),np.min(filteredXP[:,:,2]),np.min(sampledx[:,0]),np.min(filteredXP[:,:,0]))
        ylimmax = max(np.max(sampledx[:,2]),np.max(filteredXP[:,:,2]), np.min(sampledx[:,0]),np.min(filteredXP[:,:,0]))
        ##Plot 1  
        if not os.path.exists(os.path.join("Output","Experement 5")):
            os.makedirs(os.path.join("Output","Experement 5"))
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of SIS PF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[0,:,0],filteredXP[0,:,2], label="SIS PF")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","SISOut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 2
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of SIS PF with Jitter's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[1,:,0],filteredXP[1,:,2], label="SIS PF with Jitter")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","SISOutwJ"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 3
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of Generic PF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[2,:,0],filteredXP[2,:,2], label="Generic PF")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","GPFOut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 4
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of Generic PF with Jitter's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[3,:,0],filteredXP[3,:,2], label="Generic PF with Jitter")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","GPFOutwJ"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 5
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of SIR PF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[4,:,0],filteredXP[4,:,2], label="SIR PF")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","SIROut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 6
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of SIR PF with Jitter's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[5,:,0],filteredXP[5,:,2], label="SIR PF with Jitter")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","SIROutwJ"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 7
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of PF with Momentum's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[1,:,0],filteredXP[1,:,2], label="PF with Momentum")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","PFwmomentum"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 8
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of PF as described Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[7,:,0],filteredXP[7,:,2], label="SIS PF as Described")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","DescOut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
        ##Plot 9
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of EKF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXKalman[0,:,0],filteredXKalman[0,:,2], label="EKF")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","EKFOut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
        
        
    
        ##Plot 10
        if not os.path.exists(os.path.join("Output","Experement 5","ParticalDensitySIROutwJ")):
            os.makedirs(os.path.join("Output","Experement 5","ParticalDensitySIROutwJ"))
        
        for i, particals in enumerate(allParticals[5]):
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            plt.ylim((ylimmin,ylimmax))
            plt.xlim((ylimmin,ylimmax))
            axis.set_title("Plot SIR PF with Jitter's particles for t="+str(i+1))
            pts = axis.scatter(particals[:,0],particals[:,2])
            plt.savefig(os.path.join("Output","Experement 5","ParticalDensitySIROutwJ",str(i)+".pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
        
    
        ##Plot 11
        if not os.path.exists(os.path.join("Output","Experement 5","ParticalDensitySIROut")):
            os.makedirs(os.path.join("Output","Experement 5","ParticalDensitySIROut"))
        
        for i, particals in enumerate(allParticals[4]):
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            plt.ylim((ylimmin,ylimmax))
            plt.xlim((ylimmin,ylimmax))
            axis.set_title("Plot SIR PF's particles for t="+str(i+1))
            pts = axis.scatter(particals[:,0],particals[:,2])
            plt.savefig(os.path.join("Output","Experement 5","ParticalDensitySIROut",str(i)+".pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
           
    
        ##Plot 12
        if not os.path.exists(os.path.join("Output","Experement 5","EKFProbDensity")):
            os.makedirs(os.path.join("Output","Experement 5","EKFProbDensity")) 
        for i, (P,mu) in enumerate(zip(allPs[0],filteredXKalman[0])):
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel("x")
            axis.set_ylabel("p(x)")
            axis.set_title("Plot of p(x) for t="+str(i+1))
            x = mu+np.array([1,0,0,0])*np.linspace(-3*np.sqrt(P[0,0]),3*np.sqrt(P[0,0]),1000).reshape((-1,1))
            Px = multivariate_normal.pdf(x,mu,P)
            pts = axis.plot(x[:,0],Px)
            plt.savefig(os.path.join("Output","Experement 5","EKFProbDensity",str(i)+"_x.pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
        
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel(r"$\dot{x}$")
            axis.set_ylabel(r"p($\dot{x}$)")
            axis.set_title(r"Plot of p($\dot{x}$) for t="+str(i+1))
            x = mu+np.array([0,1,0,0])*np.linspace(-3*np.sqrt(P[1,1]),3*np.sqrt(P[1,1]),1000).reshape((-1,1))
            Px = multivariate_normal.pdf(x,mu,P)
            pts = axis.plot(x[:,1],Px)
            plt.savefig(os.path.join("Output","Experement 5","EKFProbDensity",str(i)+"_dotx.pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
            
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel("x")
            axis.set_ylabel("p(y)")
            axis.set_title("Plot of p(y) for t="+str(i+1))
            x = mu+np.array([0,0,1,0])*np.linspace(-3*np.sqrt(P[2,2]),3*np.sqrt(P[2,2]),1000).reshape((-1,1))
            Px = multivariate_normal.pdf(x,mu,P)
            pts = axis.plot(x[:,2],Px)
            plt.savefig(os.path.join("Output","Experement 5","EKFProbDensity",str(i)+"_y.pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
        
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel(r"$\dot{y}$")
            axis.set_ylabel(r"p($\dot{y}$)")
            axis.set_title(r"Plot of p($\dot{y}$) for t="+str(i+1))
            x = mu+np.array([0,0,0,1])*np.linspace(-3*np.sqrt(P[3,3]),3*np.sqrt(P[3,3]),1000).reshape((-1,1))
            Px = multivariate_normal.pdf(x,mu,P)
            pts = axis.plot(x[:,3],Px)
            plt.savefig(os.path.join("Output","Experement 5","EKFProbDensity",str(i)+"_doty.pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
    
        ##Plot 13
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of EKF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXKalman[0,:,0],filteredXKalman[0,:,2], label="EKF")
        pts = axis.plot(filteredXKalman[0,:,0]+2*np.sqrt(allPs[0,:,0,0]),filteredXKalman[0,:,2], label="Confidence interval", ls = "-", color = "black")
        pts = axis.plot(filteredXKalman[0,:,0]-2*np.sqrt(allPs[0,:,0,0]),filteredXKalman[0,:,2], ls = "-", color = "black")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 5","EKFOutwT"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
        
        
        ##Plot 14
        pfname = ""
        if not os.path.exists(os.path.join("Output","Experement 5","PFProbDensity")):
            os.makedirs(os.path.join("Output","Experement 5","PFProbDensity")) 
        for i, (particals, mu) in enumerate(zip(allParticals[5],filteredXP[5])):
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel("x")
            axis.set_ylabel("p(x)")
            axis.set_title("Plot of p(x) for t="+str(i+1))
            plt.hist(particals[:,0], 40)
            plt.savefig(os.path.join("Output","Experement 5","PFProbDensity",str(i)+"_x.pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
        
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel(r"$\dot{x}$")
            axis.set_ylabel(r"p($\dot{x}$)")
            axis.set_title(r"Plot of p($\dot{x}$) for t="+str(i+1))
            plt.hist(particals[:,1], 40)
            plt.savefig(os.path.join("Output","Experement 5","PFProbDensity",str(i)+"_dotx.pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
            
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel("x")
            axis.set_ylabel("p(y)")
            axis.set_title("Plot of p(y) for t="+str(i+1))
            plt.hist(particals[:,2], 40)
            plt.savefig(os.path.join("Output","Experement 5","PFProbDensity",str(i)+"_y.pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
        
            f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
            axis.set_xlabel(r"$\dot{y}$")
            axis.set_ylabel(r"p($\dot{y}$)")
            axis.set_title(r"Plot of p($\dot{y}$) for t="+str(i+1))
            plt.hist(particals[:,3], 40)
            plt.savefig(os.path.join("Output","Experement 5","PFProbDensity",str(i)+"_doty.pdf"), format='pdf', dpi=500,bbox_inches="tight")
            plt.show()
    
        
    
    if TestsToDo[4]:
        print(
        """
        Test 5: Get results of baring and squared distrance tracking problem for randdom scenarios
        """)
        numberOfParticals = 4000
        initP = 0.01*np.array([[0.5**2,        0,      0,        0],
                         [0     , 0.005**2,      0,        0],
                         [0     ,        0, 0.3**2,        0],
                         [0     ,        0,      0, 0.01**2]]) 
        initP2 = np.array([[1,        0,      0,        0],
                         [0     , 0.5**2,      0,        0],
                         [0     ,        0, 1,        0],
                         [0     ,        0,      0, 0.5**2]]) 
        
        q = 0.1**2
        Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        R = np.array([[0.005**2,0],[0,0.05**2]])
        
        problem = TestProblem4(R,Q)
        
        
        runs = 100
        errorsK = np.empty((runs, 2),dtype = np.float64)
        errorsP = np.empty((runs, 7),dtype = np.float64)
        for run in range(runs):
            
            
            initXRN = np.random.multivariate_normal([10,0,10,0], initP2)
            initX = initXRN + np.random.multivariate_normal([0,0,0,0], initP)
            initXpf = np.random.multivariate_normal(initX, initP, numberOfParticals)
        
        
            pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
            
            pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
            
            pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                                R = R, Q = Q,
                                initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
            
            
            
            ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                       ,R,Q,problem.n,problem.m)
            iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                       ,R,Q,problem.n,problem.m)
            
            sampledx, realz, noisyz = problem.generateSamples(numSamples = 24, startloc=initXRN)
            
            
            filteredXKalman, allPs, filteredXP, allParticals = evalueateFilters(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf4], initX, initP, noisyz)
            
            errorsK[run] = ((filteredXKalman - sampledx)**2).mean(axis=(1,2))
            errorsP[run] = ((filteredXP - sampledx)**2).mean(axis=(1,2))
            if run%10 == 0: 
                print(run)
        
        print("Minimum K error: ", np.min(errorsK, axis =0))
        print("LQ K error: ", np.percentile(errorsK, 25, axis=0))
        print("Med K error: ", np.percentile(errorsK, 50, axis=0))
        print("UQ K error: ", np.percentile(errorsK, 75, axis=0))
        print("Maximum K error: ", np.max(errorsK, axis =0))
        print("Mean K error: ", np.mean(errorsK, axis =0))
        print("________________________________________________________")
        print("Minimum PF error: ", np.min(errorsP, axis =0))
        print("LQ PF error: ", np.percentile(errorsP, 25, axis=0))
        print("Med PF error: ", np.percentile(errorsP, 50, axis=0))
        print("UQ PF error: ", np.percentile(errorsP, 75, axis=0))
        print("Maximum PF error: ", np.max(errorsP, axis =0))
        print("Mean PF error: ", np.mean(errorsP, axis =0))
    
    
    if TestsToDo[5]:
        print(
        """
        Test 6: Plot Graphs for variang P
        """)
        
        numberOfParticals = 4000
        initP =np.array([[1,        0,      0,        0],
                         [0     , 0.00025,      0,        0],
                         [0     ,        0, 1,        0],
                         [0     ,        0,      0, 0.01]]) 
        initP2 = np.array([[0.5**2,        0,      0,        0],
                         [0     , 0.005**2,      0,        0],
                         [0     ,        0, 0.3**2,        0],
                         [0     ,        0,      0, 0.01**2]]) 
        
        q = 0.0001**2
        Q = q*np.array([[0.25,0.5,0,0],[0.5,1,0,0],[0,0,0.25,0.5],[0,0,0.5,1]])
        R = np.array([[0.005**2]])
        
        problem = TestProblem3(R,Q)
        #initX = [0.0, 0.0, 0.4, -0.05]
        initX = [-0.05, 0.001, 0.7, -0.055]
        
        initXpf = np.random.multivariate_normal(initX, initP, numberOfParticals)
        
        
        #0.0001**2*np.eye(problem.n)
        
        
        pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
        
        pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
        
        pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
        pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=0.2)
        
        pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf, numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        pf5 = ParticalFilterAsDescribed(q, R, initXpf, numberOfParticals)
        
        
        
        ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                   ,R,Q,problem.n,problem.m)
        iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                                   ,R,Q,problem.n,problem.m)
        
        sampledx, realz, noisyz = problem.generateSamples(numSamples = 24)
        
        
        filteredXKalman, allPs, filteredXP, allParticals = evalueateFilters(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf4, pf5], initX, initP, noisyz)
       
     
        ylimmin = min(np.min(sampledx[:,2]),np.min(filteredXP[:,:,2]),np.min(sampledx[:,0]),np.min(filteredXP[:,:,0]))
        ylimmax = max(np.max(sampledx[:,2]),np.max(filteredXP[:,:,2]), np.min(sampledx[:,0]),np.min(filteredXP[:,:,0]))
        ##Plot 1  
        if not os.path.exists(os.path.join("Output","Experement 6")):
            os.makedirs(os.path.join("Output","Experement 6"))
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of SIS PF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[0,:,0],filteredXP[0,:,2], label="SIS PF")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","SISOut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 2
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of SIS PF with Jitter's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[1,:,0],filteredXP[1,:,2], label="SIS PF with Jitter")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","SISOutwJ"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 3
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of Generic PF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[2,:,0],filteredXP[2,:,2], label="Generic PF")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","GPFOut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 4
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of Generic PF with Jitter's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[3,:,0],filteredXP[3,:,2], label="Generic PF with Jitter")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","GPFOutwJ"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 5
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of SIR PF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[4,:,0],filteredXP[4,:,2], label="SIR PF")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","SIROut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 6
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of SIR PF with Jitter's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[5,:,0],filteredXP[5,:,2], label="SIR PF with Jitter")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","SIROutwJ"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 7
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of PF with Momentum's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[1,:,0],filteredXP[1,:,2], label="PF with Momentum")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","PFwmomentum"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
    
        ##Plot 8
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of PF as described Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXP[7,:,0],filteredXP[7,:,2], label="SIS PF as Described")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","DescOut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
    
        ##Plot 9
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of EKF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXKalman[0,:,0],filteredXKalman[0,:,2], label="EKF")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","EKFOut"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
        
    
           
    
        ##Plot 10
        
            
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        plt.ylim((ylimmin,ylimmax))
        plt.xlim((ylimmin,ylimmax))
        axis.set_title("Plot of EKF's Output")
        pts = axis.scatter(sampledx[:,0],sampledx[:,2], label="Actual X")
        pts = axis.scatter(filteredXKalman[0,:,0],filteredXKalman[0,:,2], label="EKF")
        pts = axis.plot(filteredXKalman[0,:,0]+2*np.sqrt(allPs[0,:,0,0]),filteredXKalman[0,:,2], label="Confidence interval", ls = "-", color = "black")
        pts = axis.plot(filteredXKalman[0,:,0]-2*np.sqrt(allPs[0,:,0,0]),filteredXKalman[0,:,2], ls = "-", color = "black")
        plt.legend()
        plt.savefig(os.path.join("Output","Experement 6","EKFOutwT"), format='pdf', dpi=500,bbox_inches="tight")
        plt.show()
        
        
    
    
        
    
