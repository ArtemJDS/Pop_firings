import numpy as np
import matplotlib.pyplot as plt


import lib

params = [{
    "MaxFR" : 1.0,
    "Sfr" : 625,
    "th" : 0.5,
    "r" : 0.1,
    "q" : 1,
    "s" : 1,
    "tau_FR" : 10,
    "tau_A" : 100,
    "winh" : 5,
},]

dt = 0.1
duration = 200
gexc = 1 + np.cos(2*np.pi*0.008*np.linspace(0, duration, int(duration/dt)+1))
ginh = 1 + np.cos(2*np.pi*0.008*np.linspace(0, duration, int(duration/dt)+1) + np.pi )
pop = lib.RateModel(params)

fr = pop.run_model(dt, duration, gexc, ginh)

plt.plot(fr)
plt.show()