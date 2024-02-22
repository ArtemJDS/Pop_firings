import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import h5py
from collections import OrderedDict
import json
import lib
import os
import logging
logging.basicConfig(filename='progress_optim.log', level=logging.DEBUG)

IS_OPTIM = True

dt = 0.1
duration = 2000

current_path = os.getcwd()
if not os.path.exists(current_path + 'optim_results'):
    os.makedirs(current_path + 'optim_results')

# path = "CA1 Axo-axonic"
paths = [i for i in os.listdir() if '_' not in i and  '.' not in i]
for ord_n, path in enumerate(paths):

    parordict = OrderedDict()
    parordict["MaxFR"] = 0.9
    parordict["Sfr"] = 625
    parordict["th"] = 0.5
    parordict["r"] = 0.1
    parordict["q"] = 0.9
    parordict["s"] = 0.9
    parordict["tau_FR"] = 10
    parordict["tau_A"] = 100
    parordict["winh"] = 5

    parameters = [parordict,]

    bounds = [
        [0.2, 1.0], # "MaxFR"
        [100, 10000], # "Sfr"
        [-100, 100], # "th"
        [0.0001, 1.0], # "r"
        [0.0001, 1.0], # "q"
        [0.0001, 1.0], # "s"
        [0.1, 100.0], # "tau_FR"
        [0.1, 1000.0], # "tau_A"
        [0.1, 100.0], # "winh"
    ]


    Niter = 100
#     path = "CA1 Axo-axonic"

    target_firing_rate = []
    gexc = []
    ginh = []

    for idx in range(Niter):
        filepath = "{path}/{i}.hdf5".format(path=path, i=idx)
        with h5py.File(filepath, mode='r') as h5file:
            gexc.append( h5file["gexc"][:].ravel() )
            ginh.append( h5file["ginh"][:].ravel() )
            target_firing_rate.append(h5file["firing_rate"][:].ravel())

 
    target_firing_rate = np.stack(target_firing_rate, axis=1)
    gexc = np.stack(gexc, axis=1)
    ginh = np.stack(ginh, axis=1)


    X = np.zeros(9, dtype=np.float64)
    for idx, val in enumerate(parordict.values()):
        X[idx] = val


    pop = lib.RateModel(Niter, parameters)

    if IS_OPTIM:
        res = differential_evolution(pop.loss, x0=X, bounds=bounds, args=(dt, target_firing_rate, gexc, ginh), \
                                        atol=1e-3, recombination=0.7, mutation=(0.7, 1.1), updating='deferred', strategy='best2bin', \
                                        disp=True, workers=-1, maxiter=700)
        print(res.message)
        print(res.x)



        for idx, key in enumerate(parordict.keys()):
            parordict[key] = res.x[idx]


        with open(f"optim_results/{path}_optim_res.json", "w") as outfile:
            json.dump(parordict, outfile)

    else:
        with open(f"optim_results/{path}_optim_res.json", "r") as outfile:
            params = json.load(outfile)

            for idx, key in enumerate(parordict.keys()):
                X[idx] = params[key]



    rate_model_firings = pop.run_from_X(X, dt, target_firing_rate.shape[0], gexc, ginh)

    t = np.linspace(0, duration, target_firing_rate.shape[0])
    for idx in range(target_firing_rate.shape[1]):
        fig, axes = plt.subplots(nrows=2)

        axes[0].set_title(idx)
        axes[0].plot(t, rate_model_firings[:, idx], label="Rate model", color="green")
        axes[0].plot(t, target_firing_rate[:, idx], label="Izhikevich model", color="red")

        axes[0].legend(loc="upper right")

        axes[1].plot(t[1:], gexc[:, idx], label="Ext conductance", color="orange")
        axes[1].plot(t[1:], ginh[:, idx], label="Inh conductance", color="blue")

        axes[1].legend(loc="upper right")
        
        plt.savefig('optim_results/' + path)
#         plt.show(block=True)

        if idx > 20:
            break
    plt.close()
#     logging.info('Created {n} %. {path}'.format(n = ord_n/len(paths) * 100, path = path))


    # optimal_params = parordict
    # file = open("res.txt", "w")
    # for key, val in optimal_params.items():
    #     file.write("{} : {}\n".format(key, val))
    # file.close()

    # gexc = 1 + np.cos(2*np.pi*0.008*np.linspace(0, duration, int(duration/dt)+1))
    # ginh = 1 + np.cos(2*np.pi*0.008*np.linspace(0, duration, int(duration/dt)+1) + np.pi )
    # fr = pop.run_model(dt, duration, gexc, ginh)
    # plt.plot(target_firing_rate[:, 55])
    # plt.show()
