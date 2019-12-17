import numpy as np
from ParticalFilter import ParticleFilter, ParticleFilterWithJitter, ParticalFilterGeneric, ParticalFilterGenericWithJitter, ParticleFilterSIR
from ParticalFilter import ParticleFilterWithWeightMomentum, ParticleFilterSIRWithJitter
from ExtendedKalmanFilter import ExtendedKalmanFilter, IteratedExtendedKalmanFilter
import os


def evalueateFiltersLessMemory(problem, KalmanFilters, ParticalFilters, initialXoutKalman, initialPK, noisyz):

    xsKalman = [initialXoutKalman.copy() for k in KalmanFilters]
    Ps = [initialPK.copy() for k in KalmanFilters]
    
    filteredXKalman = np.empty((len(KalmanFilters),len(noisyz),problem.n))
    filteredXP = np.empty((len(ParticalFilters),len(noisyz),problem.n))
        
    
    for km1, z_mes in enumerate(noisyz):
        for kfn, kf in enumerate(KalmanFilters):
            xsKalman[kfn], Ps[kfn] = kf.predict(xsKalman[kfn], Ps[kfn], km1+1)
            xsKalman[kfn], Ps[kfn], _ = kf.correction(xsKalman[kfn], Ps[kfn], z_mes, km1+1)
            filteredXKalman[kfn, km1] = xsKalman[kfn]
            
        for pfn, pf in enumerate(ParticalFilters):
            xout, _ = pf.iterate(z_mes, km1+1)
            filteredXP[pfn, km1] = xout
    return filteredXKalman, filteredXP


def evaluateAllFiltersLessMemory(problem, numSamples, initialProbX, numberOfParticals,K, R,Q, initialP, initialXmu, erroraxis):
    initXpf = np.random.multivariate_normal(initialXmu, initialP, numberOfParticals)
    
    pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
    pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
    
    pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
    
    pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)  
    
    ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                               ,R,Q,problem.n,problem.m)
    iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                               ,R,Q,problem.n,problem.m) 
    
    sampledx, realz, noisyz = problem.generateSamples(numSamples = numSamples, startloc = initialProbX)
    
    filteredXKalman, filteredXP = (
            evalueateFiltersLessMemory(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf4], initialXmu, initialP, noisyz))
    
    errorsK = ((filteredXKalman[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    errorsP = ((filteredXP[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    return errorsK, errorsP






def evaluateImportantAllFiltersLessMemory(problem, numSamples, initialProbX, numberOfParticals,K, R,Q, initialP, initialXmu, erroraxis):
    initXpf = np.random.multivariate_normal(initialXmu, initialP, numberOfParticals)
    
    pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
   
    ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                               ,R,Q,problem.n,problem.m)
    
    sampledx, realz, noisyz = problem.generateSamples(numSamples = numSamples, startloc = initialProbX)
    
    filteredXKalman, filteredXP = (
            evalueateFiltersLessMemory(problem, [ekf], [pf3j], initialXmu, initialP, noisyz))
    
    errorsK = ((filteredXKalman[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    errorsP = ((filteredXP[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    return errorsK, errorsP


def evaluatePFFilterLessMemory(problem, numSamples, initialProbX, numberOfParticals,K, R,Q, initialP, initialXmu, erroraxis):
    initXpf = np.random.multivariate_normal(initialXmu, initialP, numberOfParticals)
    
    pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    
    sampledx, realz, noisyz = problem.generateSamples(numSamples = numSamples, startloc = initialProbX)
    
    _, filteredXP = (
            evalueateFiltersLessMemory(problem, [], [pf3j], initialXmu, initialP, noisyz))
    
    errorsP = ((filteredXP[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    return errorsP





def evalueateFiltersWithWeights(problem, KalmanFilters, ParticalFilters, initialXoutKalman, initialPK, noisyz):

    xsKalman = [initialXoutKalman.copy() for k in KalmanFilters]
    Ps = [initialPK.copy() for k in KalmanFilters]
    
    filteredXKalman = np.empty((len(KalmanFilters),len(noisyz),problem.n))
    filteredXP = np.empty((len(ParticalFilters),len(noisyz),problem.n))
        
    allPs = np.empty((len(KalmanFilters),len(noisyz),problem.n,problem.n))
    allParticals = np.empty((len(ParticalFilters),len(noisyz),ParticalFilters[0].numberOfSamples,problem.n))
    allWeights = np.empty((len(ParticalFilters),len(noisyz),ParticalFilters[0].numberOfSamples))
    
    for km1, z_mes in enumerate(noisyz):
        for kfn, kf in enumerate(KalmanFilters):
            xsKalman[kfn], Ps[kfn] = kf.predict(xsKalman[kfn], Ps[kfn], km1+1)
            xsKalman[kfn], Ps[kfn], _ = kf.correction(xsKalman[kfn], Ps[kfn], z_mes, km1+1)
            filteredXKalman[kfn, km1] = xsKalman[kfn]
            allPs[kfn, km1] = Ps[kfn]
            
        for pfn, pf in enumerate(ParticalFilters):
            xout, _ = pf.iterate(z_mes, km1+1)
            filteredXP[pfn, km1] = xout
            allParticals[pfn, km1] = pf.X
            allWeights[pfn, km1] = np.exp(pf.logw)
    return filteredXKalman, allPs, filteredXP, allParticals, allWeights


def evaluateAllFiltersWithWeights(problem, numSamples, initialProbX, numberOfParticals,K, R,Q, initialP, initialXmu, erroraxis):
    initXpf = np.random.multivariate_normal(initialXmu, initialP, numberOfParticals)
    
    pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
    pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
    
    pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
    
    pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)  
    
    ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                               ,R,Q,problem.n,problem.m)
    iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                               ,R,Q,problem.n,problem.m) 
    
    sampledx, realz, noisyz = problem.generateSamples(numSamples = numSamples, startloc = initialProbX)
    
    filteredXKalman, allPs, filteredXP, allParticals, allWeights = (
            evalueateFiltersWithWeights(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf4], initialXmu, initialP, noisyz))
    
    errorsK = ((filteredXKalman[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    errorsP = ((filteredXP[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    return filteredXKalman, allPs, filteredXP, allParticals, allWeights, errorsK, errorsP, sampledx










def evalueateFilters(problem, KalmanFilters, ParticalFilters, initialXoutKalman, initialPK, noisyz):

    xsKalman = [initialXoutKalman.copy() for k in KalmanFilters]
    Ps = [initialPK.copy() for k in KalmanFilters]
    
    filteredXKalman = np.empty((len(KalmanFilters),len(noisyz),problem.n))
    filteredXP = np.empty((len(ParticalFilters),len(noisyz),problem.n))
        
    allPs = np.empty((len(KalmanFilters),len(noisyz),problem.n,problem.n))
    allParticals = np.empty((len(ParticalFilters),len(noisyz),ParticalFilters[0].numberOfSamples,problem.n))
    
    for km1, z_mes in enumerate(noisyz):
        for kfn, kf in enumerate(KalmanFilters):
            xsKalman[kfn], Ps[kfn] = kf.predict(xsKalman[kfn], Ps[kfn], km1+1)
            xsKalman[kfn], Ps[kfn], _ = kf.correction(xsKalman[kfn], Ps[kfn], z_mes, km1+1)
            filteredXKalman[kfn, km1] = xsKalman[kfn]
            allPs[kfn, km1] = Ps[kfn]
            
        for pfn, pf in enumerate(ParticalFilters):
            xout, _ = pf.iterate(z_mes, km1+1)
            filteredXP[pfn, km1] = xout
            allParticals[pfn, km1] = pf.X
    return filteredXKalman, allPs, filteredXP, allParticals


def evaluateAllFilters(problem, numSamples, initialProbX, numberOfParticals,K, R,Q, initialP, initialXmu, erroraxis):
    initXpf = np.random.multivariate_normal(initialXmu, initialP, numberOfParticals)
    
    pf1 = ParticleFilter(f = problem.bulkf, h = problem.bulkh, 
                            R = R, Q = Q,
                            initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
        
    pf1j = ParticleFilterWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf2 = ParticalFilterGeneric(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
    
    pf2j = ParticalFilterGenericWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf3 = ParticleFilterSIR(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)
    
    pf3j = ParticleFilterSIRWithJitter(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m, K=K)
    
    pf4 = ParticleFilterWithWeightMomentum(f = problem.bulkf, h = problem.bulkh, 
                        R = R, Q = Q,
                        initX = initXpf.copy(), numberOfSamples = numberOfParticals, sizeOfz = problem.m)  
    
    ekf = ExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                               ,R,Q,problem.n,problem.m)
    iekf = IteratedExtendedKalmanFilter(problem.f,problem.h,problem.Jf,problem.Jh
                               ,R,Q,problem.n,problem.m) 
    
    sampledx, realz, noisyz = problem.generateSamples(numSamples = numSamples, startloc = initialProbX)
    
    filteredXKalman, allPs, filteredXP, allParticals = (
            evalueateFilters(problem, [ekf,iekf], [pf1, pf1j, pf2, pf2j, pf3, pf3j, pf4], initialXmu, initialP, noisyz))
    
    errorsK = ((filteredXKalman[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    errorsP = ((filteredXP[:,:,erroraxis] - sampledx[:,erroraxis])**2).mean(axis=(1,2))
    return filteredXKalman, allPs, filteredXP, allParticals, errorsK, errorsP, sampledx


plotTemplate = """
    \\addplot+[
    boxplot prepared={{
    	median={median},
    	upper quartile={uq},
    	lower quartile={lq},
    	upper whisker={maxs},
    	lower whisker={mins}
    }},
    ] coordinates {{}};
    %mean is {mean}
"""


def convertToLatexBoxAndWhiscar(errorlist, path, filename, algname=None):
    if not os.path.exists(path):
        os.makedirs(path) 
        
    outputstring=plotTemplate.format(**{"median": np.percentile(errorlist, 50, axis=0),
                                        "uq":np.percentile(errorlist, 75, axis=0),
                                        "lq":np.percentile(errorlist, 25, axis=0),
                                        "maxs":np.max(errorlist, axis=0),
                                        "mins":np.min(errorlist, axis=0),
                                        "mean":np.mean(errorlist, axis=0)})
        
    if algname is not None:
        print("______Stats for "+algname+"______")
        print("Minimum error: ", np.min(errorlist, axis =0))
        print("LQ error: ", np.percentile(errorlist, 25, axis=0))
        print("Med error: ", np.percentile(errorlist, 50, axis=0))
        print("UQ error: ", np.percentile(errorlist, 75, axis=0))
        print("Maximum error: ", np.max(errorlist, axis =0))
        print("Mean error: ", np.mean(errorlist, axis =0))
        
    with open(os.path.join(path,filename),'w') as fp :
        fp.write(outputstring)    
        
        
    