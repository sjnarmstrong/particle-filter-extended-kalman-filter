import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


try:
    (_,errorMeans), (_,errorUQs), (_,errorLQs), (_,errorMedians), (_,errorMins), (_,errorMaxs), (_, Ns) = np.load(
            os.path.join("Outputs","Experement6","outputData.npz")).items()
except:
    print("Could not load data from"+os.path.join("Outputs","Experement6","outputData.npz")+".\n Please run Experement5.py before this file.")
    exit(0)
 

f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
axis.set_xlabel(r"$N_s$")
axis.set_ylabel(r"$RMSE$")
axis.set_title(r"Plot of RMSE vs $N_s$ for GPF with Jitter")


pts = axis.plot(Ns, errorLQs, label = "LQ")
pts = axis.plot(Ns, errorMedians, label = "Med")
pts = axis.plot(Ns, errorUQs, label = "UQ")

plt.legend()
plt.savefig(os.path.join("Outputs","Experement6","Stat_GPF_NS.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()
    

f, axis = plt.subplots(1, sharex=True,figsize=(15,8))
axis.set_xlabel(r"$N_s$")
axis.set_ylabel(r"$RMSE$")
axis.set_title(r"Plot of mean RMSE vs $N_s$ for GPF with Jitter")


pts = axis.plot(Ns, errorMeans)
plt.savefig(os.path.join("Outputs","Experement6","Mean_GPF_NS.pdf"), format='pdf', dpi=500,bbox_inches="tight")
plt.show()