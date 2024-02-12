import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.signal.windows import parzen
from brian2 import *
import h5py
defaultclock.dt = 0.05*ms

Cm = 114 *uF # /cm**2
k = 1.19 * mS / mV
Vrest = -57.63 *mV
Vth = -35.53   *mV
Vpeak = 21.72 *mV
Vmin = -48.7 *mV
a = 0.005 * ms**-1
b = 0.22 * mS
d = 2 * uA


Eexc = 0 *mV
Einh = -75*mV


sigma = 0.4 *mV

omega_1_e = 2 * Hz # [0.2 2]
omega_2_e = 5 * Hz # [4  12]
omega_3_e = 35 * Hz # [25  45]
omega_4_e = 75 * Hz # [50  90]


ampl_1_e = 10.0 * mS # [0.2 10]
ampl_2_e = 3.0 * mS # [0.2 10]
ampl_3_e = 2.0 * mS # [0.2 10]
ampl_4_e = 0.5 * mS # [0.2 10]


omega_1_i = 0.5 * Hz # [0.2 2]
omega_2_i = 6 * Hz # [4  12]
omega_3_i = 40 * Hz # [25  45]
omega_4_i = 70 * Hz # [50  90]


ampl_1_i = 2 * mS # [0.2 50]
ampl_2_i = 1 * mS # [0.2 50]
ampl_3_i = 0.5 * mS # [0.2 50]
ampl_4_i = 0.2 * mS # [0.2 50]




N = 4000
duration = 2000 # ms

# OLM Model
eqs = '''
dV/dt = (k*(V - Vrest)*(V - Vth) - U + Iexc + Iinh)/Cm + sigma*xi/ms**0.5 : volt
dU/dt = a * (b * (V - Vrest) - U) : ampere
Iexc = gexc*(Eexc - V)            : ampere
Iinh = ginh*(Einh - V)            : ampere
gexc = ampl_1_e*0.5*(cos(2*pi*t*omega_1_e) + 1 ) + ampl_2_e*0.5*(cos(2*pi*t*omega_2_e) + 1 ) + ampl_3_e*0.5*(cos(2*pi*t*omega_3_e) + 1 ) + ampl_4_e*0.5*(cos(2*pi*t*omega_4_e) + 1 ) : siemens
ginh = ampl_1_i*0.5*(cos(2*pi*t*omega_1_i) + 1 ) + ampl_2_i*0.5*(cos(2*pi*t*omega_2_i) + 1 ) + ampl_3_i*0.5*(cos(2*pi*t*omega_3_i) + 1 ) + ampl_4_i*0.5*(cos(2*pi*t*omega_4_i) + 1 ) : siemens
'''


neuron = NeuronGroup(N, eqs, method='heun', threshold='V > Vpeak', reset="V = Vmin; U = U + d")
neuron.V = -65 *mV
neuron.U = 0 *uA


M_full_V = StateMonitor(neuron, 'V', record=np.arange(N)[:10])
#M_full_U = StateMonitor(neuron, 'U', record=np.arange(N))
gexc_monitor = StateMonitor(neuron, 'gexc', record=0)
ginh_monitor = StateMonitor(neuron, 'ginh', record=0)

firing_monitor = SpikeMonitor(neuron)

run(duration*ms, report='text')

Varr = np.asarray(M_full_V.V/mV)

file = h5py.File('data.hdf5', mode='w')
file.create_dataset('gexc', data=np.asarray(gexc_monitor.gexc/mS).astype(np.float32))
file.create_dataset('ginh', data=np.asarray(ginh_monitor.ginh/mS).astype(np.float32))
file.close()


population_firing_rate, bins = np.histogram(firing_monitor.t / ms, range=[0, duration], bins=10*duration+1)
dbins = bins[1] - bins[0]
population_firing_rate = population_firing_rate / N / (0.001 * dbins) # spikes per second

# ###### smoothing of population firing rate #########
win = parzen(101)
win = win / np.sum(win)
population_firing_rate = np.convolve(population_firing_rate, win, mode='same')

##### plotting ####################################
fig, axes = plt.subplots(nrows=2)
axes[0].plot(bins[1:], population_firing_rate, label="Crossing the threshold")

axes[1].plot(Varr[:, :].T)
#axes[1].plot((M_full_U.U/uA).T)


plt.show()


# #plotting for visual control
# path = '/home/ivan/Data/lstm_dens/'
# Vbins = np.linspace(-90, 50, 500)
# for idx in range(0, hists.shape[1], 20):
#     fig, axes = plt.subplots(nrows=1, sharex=True)
#     axes.plot(Vbins, hists[:, idx])
#     fig.savefig(path + str(idx) + '.png', dpi=50)
#     plt.close(fig)
#
#
# plt.show()