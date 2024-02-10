import numpy as np
import matplotlib.pyplot as plt

dt = 0.5
duration = 200

Iext = np.asarray([1.5, 2.5])
tau_nu = 2
tau_a = 10
k = 2.2
k2 = 2.5
k3 = 0.01
r = 0.01

Nsteps = int(duration / dt)
Npops = 2

def I_F_cuve(x, M=1000, th=0.5, sigma=25):
    # y = 10 / (1 + np.exp(-x) )
    y = np.where(x > th, M * x**2 / (sigma**2 + x**2), 0)
    return y

nu = np.zeros([Nsteps, Npops], dtype=np.float64)


a = np.zeros_like(nu)


exp_tau_nu = np.exp(-dt/tau_nu)
exp_tau_a = np.exp(-dt/tau_a)

gmax = np.asarray([ [1.0, 1.0], [1.0, 1.0] ])
pconn = np.asarray([ [1.0, 1.0], [1.0, 1.0] ])
tau_d = np.asarray([ [10.0, 10.0], [10.0, 10.0] ])
tau_r = np.asarray([ [1.0, 1.0], [1.0, 1.0] ])
tau_f = np.asarray([ [1.0, 1.0], [1.0, 1.0] ])
Uinc = np.asarray([ [1.0, 1.0], [1.0, 1.0] ])

tau1r = np.where(tau_d != tau_r, tau_d / (tau_d - tau_r), 1e-13)

X = np.ones_like(gmax)
U = np.zeros_like(X)
R = np.zeros_like(X)

for i in range(Nsteps - 1):
    # nu[i + 1] = nu[i] + dt * (-nu[i] + (k3 - r*nu[i]) * I_F_cuve(Iext) - k2*a[i]) / tau_nu    # np.exp(-dt / )
    # a[i + 1] = a[i] + dt * (-a[i] + k * nu[i]) / tau_a

    nu_inf = (k3 - r*nu[i, :]) * I_F_cuve(Iext) - k2*a[i, :]
    a_inf = k * nu[i, :]
    nu[i + 1, :] = nu_inf - (nu_inf - nu[i, :]) * exp_tau_nu
    a[i + 1, :] = a_inf - (a_inf - a[i, :]) * exp_tau_a


plt.plot(nu)
plt.show()