import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

X0Real = np.array([-0.05, 0.001, 0.7, -0.055])

for plotaxis in range(4):
    try:
        (_,errorMeans), (_,errorUQs), (_,errorLQs), (_,errorMedians), (_, X0RealValues) = np.load(
                os.path.join("Outputs","Experement5","outputDatax_{0}.npz".format(*(plotaxis,)))).items()
    except:
        print("Could not load data from '"+os.path.join("Outputs","Experement5","outputDatax_{0}.npz".format(*(plotaxis,)))+"'.\n Please run Experement5.py before this file.")
        exit(0)
     
    
    Xaxis = np.linalg.norm(X0RealValues-X0Real,axis=1)
    f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
    axis.set_xlabel(r"$x_{0}^{{Initial}}$".format(plotaxis))
    axis.set_ylabel(r"$RMSE$".format(plotaxis))
    axis.set_title(r"Plot of RMSE vs $||x_{0}^{{Initial}} - x_{0}^{{real}}||$ for EKF".format(plotaxis))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    
    pts = axis.plot(Xaxis, errorLQs[:,0], label = "LQ")
    pts = axis.plot(Xaxis, errorMedians[:,0], label = "Med")
    pts = axis.plot(Xaxis, errorUQs[:,0], label = "UQ")
    
    plt.legend()
    plt.savefig(os.path.join("Outputs","Experement5","EKF_x_{0}.pdf".format(plotaxis)), format='pdf', dpi=500,bbox_inches="tight")
    plt.show()
    
    f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
    axis.set_xlabel(r"$||x_{0}^{{Initial}} - x_{0}^{{real}}||$".format(plotaxis))
    axis.set_ylabel(r"$RMSE$".format(plotaxis))
    axis.set_title(r"Plot of RMSE vs $||x_{0}^{{Initial}} - x_{0}^{{real}}||$ for GPF with Jitter".format(plotaxis))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    pts = axis.plot(Xaxis, errorLQs[:,1], label = "LQ")
    pts = axis.plot(Xaxis, errorMedians[:,1], label = "Med")
    pts = axis.plot(Xaxis, errorUQs[:,1], label = "UQ")
    
    plt.legend()
    plt.savefig(os.path.join("Outputs","Experement5","PF_x_{0}.pdf".format(plotaxis)), format='pdf', dpi=500,bbox_inches="tight")
    plt.show()
    
    
     
    
    Xaxis = np.linalg.norm(X0RealValues-X0Real,axis=1)
    f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
    axis.set_xlabel(r"$x_{0}^{{Initial}}$".format(plotaxis))
    axis.set_ylabel(r"$RMSE$".format(plotaxis))
    axis.set_title(r"Plot of mean RMSE vs $||x_{0}^{{Initial}} - x_{0}^{{real}}||$ for EKF".format(plotaxis))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    pts = axis.plot(Xaxis, errorMeans[:,0])
    
    plt.savefig(os.path.join("Outputs","Experement5","Stat_EKF_x_{0}.pdf".format(plotaxis)), format='pdf', dpi=500,bbox_inches="tight")
    plt.show()
    
    f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
    axis.set_xlabel(r"$||x_{0}^{{Initial}} - x_{0}^{{real}}||$".format(plotaxis))
    axis.set_ylabel(r"$RMSE$".format(plotaxis))
    axis.set_title(r"Plot of mean RMSE vs $x_{0}^{{Initial}}$ for GPF with Jitter".format(plotaxis))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    pts = axis.plot(Xaxis, errorMeans[:,1])
    
    plt.savefig(os.path.join("Outputs","Experement5","Stat_PF_x_{0}.pdf".format(plotaxis)), format='pdf', dpi=500,bbox_inches="tight")
    plt.show()