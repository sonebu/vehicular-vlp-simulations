import scipy.signal as signal
from cache.VLC_init import *
from functools import lru_cache
import numpy as np
from numba import njit

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

    # Generating the necessary signals for distance measurement algorithm.
    def gen_signals(self, f, r, N, t, delays, noise_variance):
        ### BS: adding this here since it's generic
        length_time = np.size(t);
        # necessary noise generation for signals
        noise1 = np.random.normal(0, math.sqrt(noise_variance[0]), length_time).astype('float')
        noise2 = np.random.normal(0, math.sqrt(noise_variance[1]), length_time).astype('float')
        # generating initial reference signal (original)
        s_e = np.asarray(signal.square(2 * np.pi * f * t), dtype='float')  # + (noise1 + noise2) / 2

        ### BS: numba doesn't like dictionaries since it's not LLVM-loop-optimization-friendly
        ###     and this can easily be an array, so I'll convert it to an array
        #s_r = {1: np.asarray(signal.square(2 * np.pi * f * (t + delays[0])), dtype='float') + noise1,
        #       2: np.asarray(signal.square(2 * np.pi * f * (t + delays[1])), dtype='float') + noise2  }
        s_r = np.zeros((2, length_time));
        # noise added, received signals
        s_r[0] = np.asarray(signal.square(2 * np.pi * f * (t + delays[0])), dtype='float') + noise1;
        s_r[1] = np.asarray(signal.square(2 * np.pi * f * (t + delays[1])), dtype='float') + noise2;
        # heterodyned signal, for reference
        s_h = np.asarray(signal.square(2 * np.pi * f * (r / (r + 1)) * t), dtype='float')
        # heterodyned, flip flop gate clock signal
        s_gate = np.asarray((signal.square(2 * np.pi * (f / (N * (r + 1))) * t) > 0), dtype='float');

        return s_e, s_r, s_h, s_gate

    ### BS: a numba-jit'able method can't have a "self" type inherited class definition,
    ###     that simply doesn't work in C. so we need to make this a static method,
    ###     with a name that won't conflict with any other function in the global scope.
    # Estimating the distance based on the phase difference between the sent and both received signals.
    @staticmethod
    @njit(parallel=True)
    def rtof_estimate_dist(s_e, s_r, s_h, s_gate, f, r, N, dt, t, length_time):

        # setting initial states
        #print("f:", f, "r:", r, "N:", N,"dt:",dt)
        s_clk            = np.zeros(length_time);
        s_clk_idx        = np.arange(1, length_time, 2);
        s_clk[s_clk_idx] = 1;
        
        ### BS: numba doesn't like dictionaries since it's not LLVM-loop-optimization-friendly
        ###     and this can easily be an array, so I'll convert it to an array
        #s_phi_hh         = { 1: np.zeros(np.size(t), dtype='float'), 2: np.zeros(np.size(t), dtype='float') };
        s_phi_hh = np.zeros((2, length_time));

        s_eh_state = 0

        ### BS: numba doesn't like dictionaries since it's not LLVM-loop-optimization-friendly
        ###     and this can easily be an array, so I'll convert it to an array
        #s_rh_state = {1: 0, 2: 0}
        s_rh_state = np.zeros((2))

        ### BS: numba doesn't like dictionaries since it's not LLVM-loop-optimization-friendly
        #counts = {1: [], 2: []}
        counts1 = [];
        counts2 = [];

        ### BS: numba doesn't like dictionaries since it's not LLVM-loop-optimization-friendly
        ###     and this can easily be an array, so I'll convert it to an array
        #M = {1: 0, 2: 0}
        M = np.zeros((2))
        # to check the difference between consecutive values
        s_h_diff = np.diff(s_h)

        ### BS: bu commented kod niye burda tam anlamadım
        # s_eh_states = [1 if i == 2 and j > 0 else 0 for i, j in zip(s_h_diff, s_e[1:])]
        #
        # s_rh_states = {1: [1 if i == 2 and j > 0 else 0 for i, j in zip(s_h_diff, s_r[1][1:])],
        #                 2: [1 if i == 2 and j > 0 else 0 for i, j in zip(s_h_diff, s_r[2][1:])]}

        for i in range(1, length_time):
            # detecting the falling edge
            if s_h_diff[i-1] == 2:
                # updating the states based in zero-crossing
                if s_e[i] > 0:
                    s_eh_state = 1
                else:
                    s_eh_state = 0

                if s_r[0][i] > 0:
                    s_rh_state[0] = 1
                else:
                    s_rh_state[0] = 0

                if s_r[1][i] > 0:
                    s_rh_state[1] = 1
                else:
                    s_rh_state[1] = 0
            # phase shift pulses
            s_phi_hh[0][i] = np.logical_xor(s_eh_state, s_rh_state[0]) * s_gate[i] * s_clk[i]
            s_phi_hh[1][i] = np.logical_xor(s_eh_state, s_rh_state[1]) * s_gate[i] * s_clk[i]
            # updating and incrementing based on the phase shift pulses.
            if s_gate[i] == 1:
                if s_phi_hh[0][i] == 1:
                    M[0] += 1
                if s_phi_hh[1][i] == 1:
                    M[1] += 1
                update_flag = 1
            else:
                if update_flag == 1:
                    ### BS: see above
                    # updated, reset
                    counts1.append(M[0])
                    counts2.append(M[1])
                    M[0] = 0
                    M[1] = 0
                    update_flag = 0


        
        ### BS: numba doesn't like dictionaries since it's not LLVM-loop-optimization-friendly
        ###     and it also doesn't like measuring size of growing arrays since it requires consistent
        ###     repetitive memory usage on (I guess) the heap, moving this part outsize, it's not a 
        ###     performance-related part anyhow, doesn't have to be jit'ted.  
        #fclk = 1/(2*dt)
        #dm = {'d1': ((self.c/2) * (np.asarray(counts[2]) / ((r+1) * N * fclk))),
        #      'd2': ((self.c/2) * (np.asarray(counts[1]) / ((r+1) * N * fclk)))}

        return counts1, counts2
    # Converting distance informatioin to coordinates.
    def dist_to_pos(self, dm, delays):
        #print("dm: ",dm)
        l = self.car_dist

        ### BS: numba doesn't like dictionaries since it's not LLVM-loop-optimization-friendly
        #d1 = dm['d1']
        d1 = dm[0]
        # obtaining the nearest count
        d1_err = np.abs(self.c*delays[1]/2 - d1) # since the delays are from round trips
        d1 = d1[d1_err == np.min(d1_err)][0]

        ### BS: numba doesn't like dictionaries since it's not LLVM-loop-optimization-friendly
        #d2 = dm['d2']
        d2 = dm[1];
        # obtaining the nearest count
        d2_err = np.abs(self.c*delays[0]/2 - d2)
        d2 = d2[d2_err == np.min(d2_err)][0]
        #print(d2)
        # extracting the coordinates using triangulation and distance measurements from both tx LEDs.
        y = (d2**2 - d1**2 + l**2) / (2*l)
        #print(y)
        x = -np.sqrt(d2**2 - y**2)

        return x, y
    # Calculating distances and returning coordinates from round-trip flights.
    def estimate(self, all_delays, H, noise_variance):
        # delays for d11 and d12.
        delay1 = all_delays[0][0] * 2
        delay2 = all_delays[0][1] * 2
        delays = [delay1, delay2]
        # generating signals based on the obtained delays and noise variance.
        s_e, s_r, s_h, s_gate = self.gen_signals(self.f, self.r, self.N, self.t, delays, noise_variance[0])
        # scaling the signals, related to channel gain.
        s_r[0] *= H[0][0]
        s_r[1] *= H[0][1]
        ### BS: numba and the LLVM optimizer can't really handle varying size arrays well
        ###     so we need to tell the size of the time array beforehand
        ###     also, moved dm computation outside
        length_time = np.size(self.t);
        # clock frequency
        fclk = 1/(2*self.dt)
        # obtaining the estimated distances from the sent and received signals.
        counts1, counts2 = self.rtof_estimate_dist(s_e, s_r, s_h, s_gate, self.f, self.r, self.N, self.dt, self.t, length_time)
        size_tmp = np.size(counts1) # could equivalently be counts2
        dm       = np.zeros((2,size_tmp));
        dm[0]    = ((self.c/2) * (np.asarray(counts2) / ((self.r+1) * self.N * fclk)));
        dm[1]    = ((self.c/2) * (np.asarray(counts1) / ((self.r+1) * self.N * fclk)));
        # obtaining the coordinates from measured distances.
        x1, y1 = self.dist_to_pos(dm, delays)
        # delays for d21 and d22.
        delay1 = all_delays[1][0] * 2
        delay2 = all_delays[1][1] * 2
        delays = [delay1, delay2]
        # generating signals based on the obtained delays and noise variance.
        s_e, s_r, s_h, s_gate = self.gen_signals(self.f, self.r, self.N, self.t, delays, noise_variance[1])
        # scaling the signals, related to channel gain.
        s_r[0] *= H[1][0]
        s_r[1] *= H[1][1]
        # obtaining the estimated distances from the sent and received signals.
        counts1, counts2 = self.rtof_estimate_dist(s_e, s_r, s_h, s_gate, self.f, self.r, self.N, self.dt, self.t, length_time)
        size_tmp = np.size(counts1) # could equivalently be counts2
        dm       = np.zeros((2,size_tmp));
        dm[0]    = ((self.c/2) * (np.asarray(counts2) / ((self.r+1) * self.N * fclk)));
        dm[1]    = ((self.c/2) * (np.asarray(counts1) / ((self.r+1) * self.N * fclk)));
        # obtaining the coordinates from measured distances.
        x2, y2 = self.dist_to_pos(dm, delays)
        # returning the resulting transmitter vehicle coordinates.
        tx_pos = np.array([[x1, x2], [y1, y2]])  # to obtain separate axes
        return tx_pos
