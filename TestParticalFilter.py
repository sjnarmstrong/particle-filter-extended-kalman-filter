from Problems import TestProblem1, TestProblem2
from ParticalFilter import ParticleFilter, ParticleFilterAlwaysResampling, ParticleFilterWithWeightMomentum, GenericParticalFilter
import matplotlib.pyplot as plt
import numpy as np
from ExtendedKalmanFilter import ExtendedKalmanFilter, IteratedExtendedKalmanFilter

"""
prob = TestProblem1(0.1,0.01)
pfRandStart = [-1,-1,-np.pi,-np.pi]
pfRandEnd = [1,1,np.pi,np.pi]
"""
prob = TestProblem2(10,1)
pfRandStart = [-20]
pfRandEnd = [20]
startloc = [0.1]

numSamples = 50

numParticals = 1000
particalFilterR = 10.0
particalFilterQ = 0.01

ekfR = 10.0
ekfQ = 1.0



sampledx, realz, noisyx, noisyz = prob.generateSamples(numSamples,startloc)

"""
plt.plot(np.arange(len(sampledx)), sampledx[:,1], sampledx[:,0])
plt.show()
plt.plot(np.arange(len(sampledx)), noisyx[:,1], noisyx[:,0])
plt.show()
plt.plot(np.arange(len(sampledx)), realz[:,1], realz[:,0])
plt.show()
plt.plot(np.arange(len(sampledx)), noisyz[:,1], noisyz[:,0])
plt.show()
"""

plt.plot(np.arange(len(sampledx)), sampledx[:,0])
plt.show()
plt.show()
plt.plot(np.arange(len(sampledx)), noisyx[:,0])
plt.show()
plt.plot(np.arange(len(sampledx)), realz[:,0])
plt.show()
plt.plot(np.arange(len(sampledx)), noisyz[:,0])
plt.show()


weights = [1/numParticals]*numParticals
pf = ParticleFilter(prob.bulkf,prob.bulkh,particalFilterR,particalFilterQ,
                    pfRandStart, pfRandEnd, numParticals, prob.m)
#pf = ParticleFilterWithWeightMomentum(prob.bulkf,prob.bulkh,particalFilterR,particalFilterQ,
#                    pfRandStart, pfRandEnd, numParticals, prob.m)

#pf = DumbParticalFilter(prob.bulkf,prob.bulkh,particalFilterR,particalFilterQ,
#                    pfRandStart, pfRandEnd, numParticals, prob.m)
pf = GenericParticalFilter(prob.bulkf,prob.bulkh,particalFilterR,particalFilterQ,
                    pfRandStart, pfRandEnd, numParticals, prob.m)

ekf = ExtendedKalmanFilter(prob.f,prob.h,prob.Jf,prob.Jh,ekfR,ekfQ,prob.n,prob.m)
iekf = IteratedExtendedKalmanFilter(prob.f,prob.h,prob.Jf,prob.Jh,ekfR,ekfQ,prob.n,prob.m)

xoutekf = np.zeros((prob.n))
Poutekf = np.eye(prob.n)
xoutiekf = np.zeros((prob.n))
Poutiekf = np.eye(prob.n)

filteredXpf = np.zeros((1,prob.n))
filteredXekf = xoutekf.copy().reshape((1,-1))
filteredXiekf = xoutiekf.copy().reshape((1,-1))
particalList = pf.X.copy().reshape(numParticals,prob.n)
for km1, z_mes in enumerate(noisyz):
    xout, weights = pf.iterate(z_mes,weights, km1+1)
    filteredXpf = np.append(filteredXpf, [xout],axis=0)
    
    xoutekf, Poutekf = ekf.predict(xoutekf, Poutekf, km1+1)
    xoutekf, Poutekf, _ = ekf.correction(xoutekf, Poutekf, z_mes, km1+1)
    filteredXekf = np.append(filteredXekf, [xoutekf],axis=0)
    
    xoutiekf, Poutiekf = iekf.predict(xoutiekf, Poutiekf, km1+1)
    xoutiekf, Poutiekf, _ = iekf.correction(xoutiekf, Poutiekf, z_mes, km1+1)
    filteredXiekf = np.append(filteredXiekf, [xoutiekf],axis=0)
    particalList = np.append(particalList,pf.X.reshape(numParticals,prob.n),axis=0)
   
t = np.arange(numSamples)

"""
plt.plot(t, filteredXpf[1:,1], filteredXpf[1:,0])
plt.show()
plt.plot(t, filteredXpf[1:,2], filteredXpf[1:,3])
plt.show()
plt.scatter(np.repeat(t,numParticals), particalList[numParticals:,1])
plt.show()
plt.scatter(np.repeat(t,numParticals), particalList[numParticals:,2])
plt.show()
plt.scatter(np.repeat(t,numParticals), particalList[numParticals:,3])
plt.show()
plt.scatter(np.repeat(t,numParticals), particalList[numParticals:,0],marker = ',')
plt.show()


plt.plot(t, filteredXekf[1:,1], filteredXekf[1:,0])
plt.show()
plt.plot(t, filteredXekf[1:,2], filteredXekf[1:,3])
plt.show()


plt.plot(t, filteredXiekf[1:,1], filteredXiekf[1:,0])
plt.show()
plt.plot(t, filteredXiekf[1:,2], filteredXiekf[1:,3])
plt.show()
"""

plt.plot(t, filteredXpf[1:,0])
plt.show()
plt.scatter(np.repeat(t,numParticals), particalList[numParticals:,0],marker = ',')
plt.show()


plt.plot(t, filteredXekf[1:,0])
plt.show()


plt.plot(t, filteredXiekf[1:,0])
plt.show()