from Problems import TestProblem1
import numpy as np
from Utils import evaluateAllFilters
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
figsize=(15,8)

problem = TestProblem1(0.01,0.001)
numSamples = 100
numberOfParticals = 8000
initP =np.eye(4)*0.5

filteredXKalman, allPs, filteredXP, allParticals, errorsK, errorsP, sampledx = evaluateAllFilters(problem, numSamples, [0.0,-1.0,0.0,np.pi], numberOfParticals, 0.1,
                               problem.r*np.eye(problem.m), problem.q*np.eye(problem.n), initP, [0.0, 0.0, 0.0, 0.0], [0,1])

if not os.path.exists(os.path.join("Outputs","Experement2")):
        os.makedirs(os.path.join("Outputs","Experement2"))

KtoPlot = np.arange(len(sampledx))+1
f, axis = plt.subplots(1, sharex=True,figsize=figsize)
axis.set_xlabel(r"k")
axis.set_ylabel(r"$x_0$")
axis.set_title(r"Plot of EKF's output for $x_0$")
pts = axis.plot(KtoPlot,sampledx[:,0], label=r"$x_0^{real}$", ls='-')
pts = axis.plot(KtoPlot,filteredXKalman[0,:,0], label=r"$x_0^{EKF}$")
plt.legend()
plt.savefig(os.path.join("Outputs","Experement2","EKFx0.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()

KtoPlot = np.arange(len(sampledx))+1
f, axis = plt.subplots(1, sharex=True,figsize=figsize)
axis.set_xlabel(r"k")
axis.set_ylabel(r"$x_1$")
axis.set_title(r"Plot of EKF's output for $x_1$")
pts = axis.plot(KtoPlot,sampledx[:,1], label=r"$x_1^{real}$", ls='-')
pts = axis.plot(KtoPlot,filteredXKalman[0,:,1], label=r"$x_1^{EKF}$")
plt.legend()
plt.savefig(os.path.join("Outputs","Experement2","EKFx1.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()

KtoPlot = np.arange(len(sampledx))+1
f, axis = plt.subplots(1, sharex=True,figsize=figsize)
axis.set_xlabel(r"k")
axis.set_ylabel(r"$x_2$")
axis.set_title(r"Plot of EKF's output for $x_2$")
pts = axis.plot(KtoPlot,sampledx[:,2], label=r"$x_2^{real}$", ls='-')
pts = axis.plot(KtoPlot,filteredXKalman[0,:,2], label=r"$x_2^{EKF}$")
plt.legend()
plt.savefig(os.path.join("Outputs","Experement2","EKFx2.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()

KtoPlot = np.arange(len(sampledx))+1
f, axis = plt.subplots(1, sharex=True,figsize=figsize)
axis.set_xlabel(r"k")
axis.set_ylabel(r"$x_3$")
axis.set_title(r"Plot of EKF's output for $x_3$")
pts = axis.plot(KtoPlot,sampledx[:,3], label=r"$x_3^{real}$", ls='-')
pts = axis.plot(KtoPlot,filteredXKalman[0,:,3], label=r"$x_3^{EKF}$")
plt.legend()
plt.savefig(os.path.join("Outputs","Experement2","EKFx3.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()











KtoPlot = np.arange(len(sampledx))+1
f, axis = plt.subplots(1, sharex=True,figsize=figsize)
axis.set_xlabel(r"k")
axis.set_ylabel(r"$x_0$")
axis.set_title(r"Plot of GPF with Jitter's output for $x_0$")
pts = axis.plot(KtoPlot,sampledx[:,0], label=r"$x_0^{real}$", ls='-')
pts = axis.plot(KtoPlot,filteredXP[2,:,0], label=r"$x_0^{GPFwJ}$")
plt.legend()
plt.savefig(os.path.join("Outputs","Experement2","GPFwJitterx0.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()

KtoPlot = np.arange(len(sampledx))+1
f, axis = plt.subplots(1, sharex=True,figsize=figsize)
axis.set_xlabel(r"k")
axis.set_ylabel(r"$x_1$")
axis.set_title(r"Plot of GPF with Jitter's output for $x_1$")
pts = axis.plot(KtoPlot,sampledx[:,1], label=r"$x_1^{real}$", ls='-')
pts = axis.plot(KtoPlot,filteredXP[2,:,1], label=r"$x_1^{GPFwJ}$")
plt.legend()
plt.savefig(os.path.join("Outputs","Experement2","GPFwJitterx1.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()

KtoPlot = np.arange(len(sampledx))+1
f, axis = plt.subplots(1, sharex=True,figsize=figsize)
axis.set_xlabel(r"k")
axis.set_ylabel(r"$x_2$")
axis.set_title(r"Plot of GPF with Jitter's output for $x_2$")
pts = axis.plot(KtoPlot,sampledx[:,2], label=r"$x_2^{real}$", ls='-')
pts = axis.plot(KtoPlot,filteredXP[2,:,2], label=r"$x_2^{GPFwJ}$")
plt.legend()
plt.savefig(os.path.join("Outputs","Experement2","GPFwJitterx2.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()

KtoPlot = np.arange(len(sampledx))+1
f, axis = plt.subplots(1, sharex=True,figsize=figsize)
axis.set_xlabel(r"k")
axis.set_ylabel(r"$x_3$")
axis.set_title(r"Plot of GPF with Jitter's output for $x_3$")
pts = axis.plot(KtoPlot,sampledx[:,3], label=r"$x_3^{real}$", ls='-')
pts = axis.plot(KtoPlot,filteredXP[2,:,3], label=r"$x_3^{GPFwJ}$")
plt.legend()
plt.savefig(os.path.join("Outputs","Experement2","GPFwJitterx3.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()