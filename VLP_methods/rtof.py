import scipy.signal as signal
from cache.VLC_init import *
from functools import lru_cache
import numpy as np
import numba

# import module

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


class RToF:

    def __init__(self, a_m=2, f_m=1e6, measure_dt=5e-9, vehicle_dt=5e-3, car_dist=1.6, r=499, N=1, c=3e8):
        self.dt = measure_dt
        self.f = f_m
        self.a_m = a_m
        self.r = r
        self.N = N
        self.c = c
        self.t = np.arange(0, vehicle_dt - self.dt, self.dt)
        self.car_dist = car_dist

    def gen_signals(self, f, r, N, t, delays, noise_variance):
        # TODO: s_e generation noise???
        noise1 = np.random.normal(0, math.sqrt(noise_variance[0]), np.size(t))
        noise2 = np.random.normal(0, math.sqrt(noise_variance[1]), np.size(t))

        s_e = signal.square(2 * np.pi * f * t)  # + (noise1 + noise2) / 2
        s_r = {1: signal.square(2 * np.pi * f * (t + delays[0])) + noise1,
               2: signal.square(2 * np.pi * f * (t + delays[1])) + noise2}
        s_h = signal.square(2 * np.pi * f * (r / (r + 1)) * t)

        return s_e, s_r, s_h
    @numba.jit(fastmath=True, parallel=True)
    def estimate_dist(self, s_e, s_r, s_h, f, r, N, dt, t):
        #print("f:", f, "r:", r, "N:", N,"dt:",dt)
        s_gate = (signal.square(2 * np.pi * (f / (N * (r + 1))) * t) > 0)
        s_clk = np.zeros(np.size(t))
        s_clk[np.arange(1, np.size(s_clk), 2)] = 1
        s_phi_hh = {1: np.zeros(np.size(t)), 2: np.zeros(np.size(t))}

        s_eh_state = 0
        s_rh_state = {1: 0, 2: 0}

        counts = {1: [], 2: []}
        M = {1: 0, 2: 0}

        s_h_diff = np.diff(s_h)
        # s_eh_states = [1 if i == 2 and j > 0 else 0 for i, j in zip(s_h_diff, s_e[1:])]
        #
        # s_rh_states = {1: [1 if i == 2 and j > 0 else 0 for i, j in zip(s_h_diff, s_r[1][1:])],
        #                 2: [1 if i == 2 and j > 0 else 0 for i, j in zip(s_h_diff, s_r[2][1:])]}
        for i in range(1, np.size(t)):

            if s_h_diff[i-1] == 2:

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

            s_phi_hh[1][i] = np.logical_xor(s_eh_state, s_rh_state[1]) * s_gate[i] * s_clk[i]
            s_phi_hh[2][i] = np.logical_xor(s_eh_state, s_rh_state[2]) * s_gate[i] * s_clk[i]

            if s_gate[i] == 1:
                if s_phi_hh[1][i] == 1:
                    M[1] += 1
                if s_phi_hh[2][i] == 1:
                    M[2] += 1
                update_flag = 1
            else:
                if update_flag == 1:
                    counts[1].append(M[1])
                    counts[2].append(M[2])
                    M[1] = 0
                    M[2] = 0
                    update_flag = 0

        fclk = 1/(2*dt)
        dm = {'d1': ((self.c/2) * (np.asarray(counts[2]) / ((r+1) * N * fclk))),
              'd2': ((self.c/2) * (np.asarray(counts[1]) / ((r+1) * N * fclk)))}

        return dm

    def dist_to_pos(self, dm, delays):
        #print("dm: ",dm)
        l = self.car_dist
        d1 = dm['d1']
        d1_err = np.abs(self.c*delays[1]/2 - d1) # since the delays are from round trips
        d1 = d1[d1_err == np.min(d1_err)][0]
        d2 = dm['d2']
        d2_err = np.abs(self.c*delays[0]/2 - d2)
        d2 = d2[d2_err == np.min(d2_err)][0]
        #print(d2)
        y = (d2**2 - d1**2 + l**2) / (2*l)
        #print(y)
        x = -np.sqrt(d2**2 - y**2)

        return x, y
    @numba.jit(fastmath=True, parallel=True)
    def estimate(self, all_delays, H, noise_variance):
        
        delay1 = all_delays[0][0] * 2
        delay2 = all_delays[0][1] * 2

        delays = [delay1, delay2]
        s_e, s_r, s_h = self.gen_signals(self.f, self.r, self.N, self.t, delays, noise_variance[0])

        s_r[1] *= H[0][0]
        s_r[2] *= H[0][1]
        dm = self.estimate_dist(s_e, s_r, s_h, self.f, self.r, self.N, self.dt, self.t)

        x1, y1 = self.dist_to_pos(dm, delays)

        delay1 = all_delays[1][0] * 2
        delay2 = all_delays[1][1] * 2

        delays = [delay1, delay2]
        s_e, s_r, s_h = self.gen_signals(self.f, self.r, self.N, self.t, delays, noise_variance[1])
        # channel attenuation
        s_r[1] *= H[1][0]
        s_r[2] *= H[1][1]
        dm = self.estimate_dist(s_e, s_r, s_h, self.f, self.r, self.N, self.dt, self.t)

        x2, y2 = self.dist_to_pos(dm, delays)

        tx_pos = np.array([[x1, x2], [y1, y2]])  # to obtain separate axes
        return tx_pos
