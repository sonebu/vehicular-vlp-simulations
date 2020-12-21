import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from VLC_init import *
# import module
import cProfile

"""
*: coordinate center of cari

|--------|                     
| car1   |                      
|-------*|
         |
       y |                   |---------|
         |                   |  car2   |
         |-------------------|*--------|
                    d

"""


class RToF_pos:

    def __init__(self, vlc_obj):

        self.vlc_obj = vlc_obj
        self.dt = 5e-9
        self.f = 1e6
        self.r = 499
        self.N = 1
        self.t = np.arange(0, 5e-3 - self.dt, self.dt)

    def gen_signals(self, f, r, N, t, delays):

        noise = np.random.normal(0, 1e-1, np.size(t))

        s_e = signal.square(2 * np.pi * f * t) + noise
        s_r = {1: signal.square(2 * np.pi * f * (t + delays[0])) + noise,
               2: signal.square(2 * np.pi * f * (t + delays[1])) + noise}
        s_h = signal.square(2 * np.pi * f * (r / (r + 1)) * t)

        return s_e, s_r, s_h

    def estimate_dist(self, s_e, s_r, s_h, f, r, N, dt, t):

        s_gate = (signal.square(2 * np.pi * (f / (N * (r + 1))) * t) > 0)
        s_clk = np.zeros(np.size(t))
        s_clk[np.arange(1, np.size(s_clk), 2)] = 1

        s_eh = np.zeros(np.size(t))
        s_rh = {1: np.zeros(np.size(t)), 2: np.zeros(np.size(t))}
        s_phi = {1: np.zeros(np.size(t)), 2: np.zeros(np.size(t))}
        s_phi_h = {1: np.zeros(np.size(t)), 2: np.zeros(np.size(t))}
        s_phi_hh = {1: np.zeros(np.size(t)), 2: np.zeros(np.size(t))}

        s_eh_state = 0
        s_rh_state = {1: 0, 2: 0}

        counts = {1: [], 2: []}
        M = {1: 0, 2: 0}

        for i in range(1, np.size(t)):

            if s_h[i] - s_h[i - 1] == 2:

                if s_e[i] > 0:
                    s_eh_state = 1
                else:
                    s_eh_state = 0

                if s_r[1][i] > 0:
                    s_rh_state[1] = 1
                else:
                    s_rh_state[1] = 0

                if s_r[2][i] > 0:
                    s_rh_state[2] = 1
                else:
                    s_rh_state[2] = 0

            s_eh[i] = s_eh_state
            s_rh[1][i] = s_rh_state[1]
            s_rh[2][i] = s_rh_state[2]

            s_phi[1][i] = np.logical_xor(s_eh_state, s_rh_state[1])
            s_phi[2][i] = np.logical_xor(s_eh_state, s_rh_state[2])

            s_phi_h[1][i] = s_phi[1][i] * s_gate[i]
            s_phi_h[2][i] = s_phi[2][i] * s_gate[i]

            s_phi_hh[1][i] = s_phi_h[1][i] * s_clk[i]
            s_phi_hh[2][i] = s_phi_h[2][i] * s_clk[i]

            if s_gate[i] == 1:
                if s_phi_hh[1][i] == 1:
                    M[1] += 1
                if (s_phi_hh[2][i] == 1):
                    M[2] += 1
                update_flag = 1
            else:
                if (update_flag == 1):
                    counts[1].append(M[1])
                    counts[2].append(M[2])
                    M[1] = 0
                    M[2] = 0
                    update_flag = 0

        fclk = 1 / (2 * dt)
        dm = {'dr': ((self.vlc_obj.c / 2) * (np.asarray(counts[1]) / ((r + 1) * N * fclk))),
              'dl': ((self.vlc_obj.c / 2) * (np.asarray(counts[2]) / ((r + 1) * N * fclk)))}

        return dm

    def dist_to_pos(self, dm, delays):
        l = self.vlc_obj.distancecar
        dr = dm['dr']
        dr_err = np.abs(self.vlc_obj.c * delays[0] / 2 - dr)  # since the delays are from round trips
        dr = dr[dr_err == np.min(dr_err)][0]
        dl = dm['dl']
        dl_err = np.abs(self.vlc_obj.c * delays[1] / 2 - dl)
        dl = dl[dl_err == np.min(dl_err)][0]

        y = (dl ** 2 - dr ** 2 + l ** 2) / (2 * l)
        x = -np.sqrt(dl ** 2 - y ** 2)

        return x, y

    def estimate(self):

        delay1 = self.vlc_obj.delays[0][1] * 2
        delay2 = self.vlc_obj.delays[1][1] * 2

        delays = [delay1, delay2]
        s_e, s_r, s_h = self.gen_signals(self.f, self.r, self.N, self.t, delays)

        s_r[1] *= self.vlc_obj.H[0][1]
        s_r[2] *= self.vlc_obj.H[1][1]
        dm = self.estimate_dist(s_e, s_r, s_h, self.f, self.r, self.N, self.dt, self.t)

        x1, y1 = self.dist_to_pos(dm, delays)

        delay1 = self.vlc_obj.delays[0][0] * 2
        delay2 = self.vlc_obj.delays[1][0] * 2

        delays = [delay1, delay2]
        s_e, s_r, s_h = self.gen_signals(self.f, self.r, self.N, self.t, delays)
        # channel attenuation
        s_r[1] *= self.vlc_obj.H[0][0]
        s_r[2] *= self.vlc_obj.H[1][0]
        dm = self.estimate_dist(s_e, s_r, s_h, self.f, self.r, self.N, self.dt, self.t)

        x2, y2 = self.dist_to_pos(dm, delays)

        tx_pos = np.array([[x1, x2], [y1, y2]])  # to obtain separate axes
        return tx_pos
