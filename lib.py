import numpy as np


class RateModel:

    def __init__(self, params):
        self.MaxFR = np.zeros(len(params), dtype=np.float64)
        self.Sfr = np.zeros(len(params), dtype=np.float64)
        self.th = np.zeros(len(params), dtype=np.float64)
        self.r = np.zeros(len(params), dtype=np.float64)
        self.s = np.zeros(len(params), dtype=np.float64)
        self.q = np.zeros(len(params), dtype=np.float64)
        self.tau_FR = np.zeros(len(params), dtype=np.float64)
        self.tau_A = np.zeros(len(params), dtype=np.float64)
        self.winh = np.zeros(len(params), dtype=np.float64)

        for idx, p in enumerate(params):
            self.MaxFR[idx] = p["MaxFR"]
            self.Sfr[idx] = p["Sfr"]
            self.th[idx] = p["th"]

            self.r[idx] = p["r"]
            self.q[idx] = p["q"]
            self.s[idx] = p["s"]

            self.tau_FR[idx] = p["tau_FR"]
            self.tau_A[idx] = p["tau_A"]
            self.winh[idx] = p["winh"]

    def I_F_cuve(self, x):
        shx_2 = (x - self.th)**2
        y = np.where(x > self.th, self.MaxFR * shx_2 / (self.Sfr + shx_2), 0)
        return y

    def run_model(self, dt, duration, gexc, ginh):
        Nsteps = int(duration / dt)
        Npops = self.tau_A.size

        FR = np.zeros([Nsteps, Npops], dtype=np.float64)

        Ad = np.zeros_like(FR)

        exp_tau_FR = np.exp(-dt / self.tau_FR)
        exp_tau_A = np.exp(-dt / self.tau_A)

        for i in range(Nsteps - 1):
            FR_inf = (1 - self.r * FR[i, :]) * self.I_F_cuve(gexc[i] - self.winh * ginh[i] - self.q * Ad[i, :])
            A_inf = self.s * FR[i, :]


            FR[i + 1, :] = FR_inf - (FR_inf - FR[i, :]) * exp_tau_FR
            Ad[i + 1, :] = A_inf - (A_inf - Ad[i, :]) * exp_tau_A


        return FR