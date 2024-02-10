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

def I_F_cuve(x, M=1000, th=1.5, sigma=25):
    # y = 10 / (1 + np.exp(-x) )
    y = np.where(x > th, M * x**2 / (sigma**2 + x**2), 0)
    return y

nu = np.zeros([Nsteps, Npops], dtype=np.float64)


a = np.zeros_like(nu)


exp_tau_nu = np.exp(-dt/tau_nu)
exp_tau_a = np.exp(-dt/tau_a)

gmax = 10 * np.asarray([ [1.0, 10.1], [-50.0, 0.0] ])
pconn = np.asarray([ [1.0, 1.0], [1.0, 1.0] ])
tau_d = np.asarray([ [750.0, 750.0], [750.0, 750.0] ])
tau_r = np.asarray([ [20.0, 20.0], [20.0, 20.0] ])
tau_f = np.asarray([ [50.0, 50.0], [50.0, 50.0] ])
Uinc = np.asarray([ [0.45, 0.45], [0.45, 0.45] ])

tau1r = np.where(tau_d != tau_r, tau_d / (tau_d - tau_r), 1e-13)

X = np.ones_like(gmax)
U = np.zeros_like(X)
R = np.zeros_like(X)

Isyn = np.zeros(Npops, dtype=np.float64)

#gmax = np.asarray([ [0.0, 1.0], [-1.0, 0.0]])


for i in range(Nsteps - 1):
    ## nu[i + 1] = nu[i] + dt * (-nu[i] + (k3 - r*nu[i]) * I_F_cuve(Iext) - k2*a[i]) / tau_nu    # np.exp(-dt / )
    ## a[i + 1] = a[i] + dt * (-a[i] + k * nu[i]) / tau_a

    nu_inf = (k3 - r*nu[i, :]) * I_F_cuve(Isyn + Iext) - k2*a[i, :]
    a_inf = k * nu[i, :]
    nu[i + 1, :] = nu_inf - (nu_inf - nu[i, :]) * exp_tau_nu
    a[i + 1, :] = a_inf - (a_inf - a[i, :]) * exp_tau_a

    Spre_normed = nu[i, :].reshape(-1, 1) * pconn
    #Spre_normed = np.asarray([1, 2]).reshape(-1, 1) * pconn


    # print(Spre_normed)
    #print(np.sum(Spre_normed, axis=))


    y_ = R * np.exp(-dt / tau_d)

    x_ = 1 + (X - 1 + tau1r * U) * np.exp(-dt / tau_r) - tau1r * U

    u_ = U * np.exp(-dt / tau_f)
    U = u_ + Uinc * (1 - u_) * Spre_normed
    R = y_ + U * x_ * Spre_normed
    X = x_ - U * x_ * Spre_normed

    Isyn = np.sum(gmax*R, axis=0)

    # Isyn = np.sum(gmax * Spre_normed, axis=0)
    # print(Isyn)
    # break


plt.plot(nu)
plt.show()