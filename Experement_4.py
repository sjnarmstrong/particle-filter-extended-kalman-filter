from Problems import TestProblem3
import numpy as np
from Utils import evaluateAllFiltersWithWeights
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
plt.rcParams.update({'font.size': 20})

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

filteredXKalman, allPs, filteredXP, allParticals, allWeights, errorsK, errorsP, sampledx = evaluateAllFiltersWithWeights(problem, numSamples, X0Real, numberOfParticals, K,
                               R, Q, initP, initX, [0,1,2,3])

if not os.path.exists(os.path.join("Outputs","Experement4")):
        os.makedirs(os.path.join("Outputs","Experement4"))
        

for i, (P, muekf, particals, weights, mupf, actualx) in enumerate(zip(allPs[0],filteredXKalman[0],allParticals[2], allWeights[2],filteredXP[2],sampledx)):

    
    for plotaxis in range(4):
        f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
        axis.set_xlabel(r"$x_{0}$".format(plotaxis))
        axis.set_ylabel(r"Prior likelihood of $x_{0}$".format(plotaxis))
        axis.set_title("Plots of Likelihoods for EKF and PF at t="+str(i+1))
        
        #weights = np.ones_like(particals[:,0])/float(len(particals[:,0]))
        barheights, bins, _ = plt.hist(particals[:,plotaxis], 50, weights=weights, label = "GPF with Jitter")
        
        xkf = muekf+np.array(np.eye(4)[plotaxis])*np.linspace(min(-3*np.sqrt(P[plotaxis,plotaxis]),bins[0]-muekf[plotaxis]),
                             max(3*np.sqrt(P[plotaxis,plotaxis]),bins[-1]-muekf[plotaxis]),1000).reshape((-1,1))
        Pxkf = multivariate_normal.pdf(xkf,muekf,P)
        Pxkf *= np.max(barheights)/np.max(Pxkf)
        
        
        pts = axis.plot(xkf[:,plotaxis],Pxkf, label = "EKF")
        plt.axvline(x=actualx[plotaxis], ls='--', color = 'r', label = r"$x_{0}^{{Real}}$".format(plotaxis))
        plt.axvline(x=muekf[plotaxis], ls='--', color = 'm', label = r"$x_{0}^{{EKF}}$".format(plotaxis))
        plt.axvline(x=mupf[plotaxis], ls='--', color = 'g', label = r"$x_{0}^{{PF}}$".format(plotaxis))
        plt.legend()
        plt.savefig(os.path.join("Outputs","Experement4","x"+str(plotaxis)+"_"+str(i)+"_x.pdf"), format='pdf', dpi=500,bbox_inches="tight")
        #plt.show()
        plt.close()
    
 