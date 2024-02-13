import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import h5py
import lib

dt = 0.1
duration = 200

params = [{
    "MaxFR" : 0.9,
    "Sfr" : 625,
    "th" : 0.5,
    "r" : 0.1,
    "q" : 0.9,
    "s" : 0.9,
    "tau_FR" : 10,
    "tau_A" : 100,
    "winh" : 5,
},]

bounds = [
    [0.2, 1.0], # "MaxFR"
    [100, 10000], # "Sfr"
    [0, 100], # "th"
    [0.0001, 1.0], # "r"
    [0.0001, 1.0], # "q"
    [0.0001, 1.0], # "s"
    [0.1, 100.0], # "tau_FR"
    [0.1, 1000.0], # "tau_A"
    [0.1, 100.0], # "winh"
]

with h5py.File("data.hdf5", mode='r') as h5file:
    gexc = h5file["gexc"][:].ravel()
    ginh = h5file["ginh"][:].ravel()
    target_firing_rate = h5file["firing_rate"][:].ravel()


pop = lib.RateModel(params)
res = differential_evolution(pop.loss, bounds, args=(dt, target_firing_rate, gexc, ginh), \
                             disp=True, maxiter=10)

print(res.x)

# gexc = 1 + np.cos(2*np.pi*0.008*np.linspace(0, duration, int(duration/dt)+1))
# ginh = 1 + np.cos(2*np.pi*0.008*np.linspace(0, duration, int(duration/dt)+1) + np.pi )
# fr = pop.run_model(dt, duration, gexc, ginh)
# plt.plot(fr)
# plt.show()