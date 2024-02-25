import numpy as np


class RateModel:

    def __init__(self, Npops, params):
        self.Npops = Npops

        self.MaxFR = np.zeros(1, dtype=np.float64)
        self.Sfr = np.zeros(1, dtype=np.float64)
        self.th = np.zeros(1, dtype=np.float64)
        self.r = np.zeros(1, dtype=np.float64)
        self.s = np.zeros(1, dtype=np.float64)
        self.q = np.zeros(1, dtype=np.float64)
        self.tau_FR = np.zeros(1, dtype=np.float64)
        self.tau_A = np.zeros(1, dtype=np.float64)
        self.winh = np.zeros(1, dtype=np.float64)

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

            break

    def I_F_cuve(self, x):
        shx_2 = (x - self.th)**2
        y = np.where(x > 0, self.MaxFR * shx_2 / (self.Sfr + shx_2), 0)
        return y

    def run_model(self, dt, Nsteps, gexc, ginh):

        FR = np.zeros([Nsteps, self.Npops], dtype=np.float64)

        Ad = np.zeros_like(FR)

        exp_tau_FR = np.exp(-dt / self.tau_FR)
        exp_tau_A = np.exp(-dt / self.tau_A)

        for i in range(Nsteps - 1):
            FR_inf = (1 - self.r * FR[i, :]) * self.I_F_cuve(gexc[i, :] - self.winh * ginh[i, :] - self.q * Ad[i, :] + self.th)
            A_inf = self.s * FR[i, :]


            FR[i + 1, :] = FR_inf - (FR_inf - FR[i, :]) * exp_tau_FR
            Ad[i + 1, :] = A_inf - (A_inf - Ad[i, :]) * exp_tau_A



        return FR

    def run_from_X(self, X, dt, Nsteps, gexc, ginh):
        self.MaxFR[0] = X[0]
        self.Sfr[0] = X[1]
        self.th[0] = X[2]

        self.r[0] = X[3]
        self.q[0] = X[4]
        self.s[0] = X[5]

        self.tau_FR[0] = X[6]
        self.tau_A[0] = X[7]
        self.winh[0] = X[8]



        FRsim = self.run_model(dt, Nsteps, gexc, ginh)
        return FRsim

    def loss(self, X, dt, FRtar, gexc, ginh):
        Nsteps = FRtar.shape[0]
        FRsim = self.run_from_X(X, dt, Nsteps, gexc, ginh)
        L = np.sum( np.log( (FRsim+1.0)/(FRtar+1) )**2 )

        return L