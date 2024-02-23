import pandas as pd
from brian2 import *
from brian2.units.allunits import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.signal.windows import parzen
from brian2 import *
from brian2.units.allunits import *
import h5py
import os
import logging
logging.basicConfig(filename='progress.log', level=logging.DEBUG)

NN = 4000



def randinterval(minv, maxv):
    v = float( (maxv - minv) * np.random.rand() + minv)
    return v


def add_units(value, key):
    if key == "Cm":
        return value* uF
    if key == "k":
        return value* mS / mV
    if key == "Vrest":
        return value* mV
    if key == "Vth":
        return np.random.normal(loc=value, scale=4.0, size=NN)*mV
    if key == "Vpeak":
        return value* mV
    if key == "Vmin":
        return value* mV
    if key == "a":
        return value * ms**-1
    if key == "b":
        return value* mS
    if key == "d":
        return value* uA

    
default_params = {
    "Cm" : 114 *uF, # /cm**2,
    "k" : 1.19 * mS / mV,
    "Vrest" : -57.63 *mV,
    "Vth" : np.random.normal(loc=-35.53, scale=4.0, size=NN)*mV,  # -35.53*mV,
    "Vpeak" : 21.72 *mV,
    "Vmin" : -48.7 *mV,
    "a" : 0.005 * ms**-1,
    "b" : 0.22 * mS,
    "d" : 2 * uA,

    "Eexc" : 0 *mV,
    "Einh" : -75*mV,


    "sigma" : 0.4 *mV,

    "omega_1_e" : randinterval(0.2, 2.0) * Hz, # [0.2 2],
    "omega_2_e" : randinterval(4.0, 12.0) * Hz, # [4  12],
    "omega_3_e" : randinterval(25.0, 45.0) * Hz, # [25  45],
    "omega_4_e" : randinterval(50.0, 90.0) * Hz, # [50  90],

    "ampl_1_e" : randinterval(10.0, 20.0) * mS, # [0.2 10],
    "ampl_2_e" : randinterval(10.0, 20.0) * mS, # [0.2 10],
    "ampl_3_e" : randinterval(10.0, 20.0) * mS, # [0.2 10],
    "ampl_4_e" : randinterval(10.0, 20.0) * mS, # [0.2 10],

    "phase0_1_e": randinterval(-np.pi, np.pi),  # [-pi pi],
    "phase0_2_e": randinterval(-np.pi, np.pi),  #  [-pi pi],
    "phase0_3_e": randinterval(-np.pi, np.pi),  #  [-pi pi],
    "phase0_4_e": randinterval(-np.pi, np.pi),  #  [-pi pi],


    "omega_1_i" : randinterval(0.2, 2.0) * Hz, # [0.2 2],
    "omega_2_i" : randinterval(4.0, 12.0) * Hz, # [4  12],
    "omega_3_i" : randinterval(25.0, 45.0) * Hz, # [25  45],
    "omega_4_i" : randinterval(50.0, 90.0) * Hz, # [50  90],


    "ampl_1_i" : randinterval(5.0, 50.0) * mS, # [0.2 50],
    "ampl_2_i" : randinterval(5.0, 50.0) * mS, # [0.2 50],
    "ampl_3_i" : randinterval(5.0, 50.0) * mS, # [0.2 50],
    "ampl_4_i" : randinterval(5.0, 50.0) * mS, # [0.2 50]

    "phase0_1_i": randinterval(-np.pi, np.pi),  # [-pi pi],
    "phase0_2_i": randinterval(-np.pi, np.pi),  # [-pi pi],
    "phase0_3_i": randinterval(-np.pi, np.pi),  # [-pi pi],
    "phase0_4_i": randinterval(-np.pi, np.pi),  # [-pi pi],

}


filepath = 'izhikevich_model_params.csv'
syndata = pd.read_csv(filepath, index_col=0)
syndata = syndata.fillna(-1)
syndata = syndata.astype(float)
syndata = syndata.rename(columns={'Vr':'Vrest', 'Vt':'Vth', 'C':'Cm'})

print(syndata.head())


all_params = {}
for neuron_type in syndata.index:
    
    neuron = syndata.loc[neuron_type]
    if not -1. in  list(neuron):
        
        neuron_opt_params = default_params.copy()

        neuron = neuron.to_dict()

        for key,item in neuron.items():
            neuron_opt_params[key] = add_units(item, key)
        all_params[neuron_type] = neuron_opt_params
    
    
def randinterval(minv, maxv):
    v = float( (maxv - minv) * np.random.rand() + minv)
    return v
def run_izhikevich_neurons(params, duration, N, filepath):
    eqs = '''
    dV/dt = (k*(V - Vrest)*(V - Vth) - U + Iexc + Iinh)/Cm + sigma*xi/ms**0.5 : volt
    dU/dt = a * (b * (V - Vrest) - U) : ampere
    Iexc = gexc*(Eexc - V)            : ampere
    Iinh = ginh*(Einh - V)            : ampere
    gexc = ampl_1_e*0.5*(cos(2*pi*t*omega_1_e + phase0_1_e) + 1 ) + ampl_2_e*0.5*(cos(2*pi*t*omega_2_e + phase0_2_e) + 1 ) + ampl_3_e*0.5*(cos(2*pi*t*omega_3_e + phase0_3_e) + 1 ) + ampl_4_e*0.5*(cos(2*pi*t*omega_4_e + phase0_4_e) + 1 ) : siemens
    ginh = ampl_1_i*0.5*(cos(2*pi*t*omega_1_i + phase0_1_i) + 1 ) + ampl_2_i*0.5*(cos(2*pi*t*omega_2_i + phase0_2_i) + 1 ) + ampl_3_i*0.5*(cos(2*pi*t*omega_3_i + phase0_3_i) + 1 ) + ampl_4_i*0.5*(cos(2*pi*t*omega_4_i + phase0_4_i) + 1 ) : siemens
    Vth : volt
    '''


    neuron = NeuronGroup(N, eqs, method='heun', threshold='V > Vpeak', reset="V = Vmin; U = U + d", namespace=params)
    neuron.V = -65 *mV
    neuron.U = 0 *uA
    neuron.Vth = params['Vth']


    M_full_V = StateMonitor(neuron, 'V', record=np.arange(N)[:10])
    #M_full_U = StateMonitor(neuron, 'U', record=np.arange(N))
    gexc_monitor = StateMonitor(neuron, 'gexc', record=0)
    ginh_monitor = StateMonitor(neuron, 'ginh', record=0)

    firing_monitor = SpikeMonitor(neuron)

    monitors = [M_full_V, gexc_monitor, ginh_monitor, firing_monitor]

    net = Network(neuron)  # automatically include G and S
    net.add(monitors)  # manually add the monitors

    net.run(duration*ms, report='text')

    Varr = np.asarray(M_full_V.V/mV)

    population_firing_rate, bins = np.histogram(firing_monitor.t / ms, range=[0, duration], bins=int(10*duration + 1) )
    #dbins = bins[1] - bins[0]
    population_firing_rate = population_firing_rate / N # spikes in bin   #/ (0.001 * dbins) # spikes per second

    # ###### smoothing of population firing rate #########
    win = parzen(101)
    win = win / np.sum(win)
    population_firing_rate = np.convolve(population_firing_rate, win, mode='same')

    file = h5py.File(filepath, mode='w')
    file.create_dataset('firing_i', data=np.asarray(firing_monitor.i).astype(np.float32))
    file.create_dataset('firing_t', data=np.asarray(firing_monitor.t / ms).astype(np.float32))
    file.create_dataset('firing_rate', data=population_firing_rate.astype(np.float32))
    file.create_dataset('gexc', data=np.asarray(gexc_monitor.gexc / mS).astype(np.float32))
    file.create_dataset('ginh', data=np.asarray(ginh_monitor.ginh / mS).astype(np.float32))
    file.close()



def create_single_type_dataset(params, path, Niter = 120, duration = 2000, NN = NN):
    
    for idx in range(Niter):
        
        filepath = '{path}/{i}.hdf5'.format(path=path, i=idx)
        run_izhikevich_neurons(params, duration, NN, filepath)


def create_all_types_dataset(all_params):
    for n,( key, item) in enumerate(all_params.items()):
        if key not in ['CA1 Oriens/Alveus']:
            params = item


            path = './{key}'.format(key = key)
    #         import shutil
    #         shutil.rmtree(path)
            os.mkdir('./{key}'.format(key = key))
            path = './{key}'.format(key = key)


            create_single_type_dataset(params, path)
            logging.info('Created {n} %'.format(n = n/len(all_params) * 100))
        
    
if __name__ == '__main__':
    create_all_types_dataset(all_params)