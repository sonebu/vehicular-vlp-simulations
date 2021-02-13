import pickle

import scipy.signal as signal
from Bound_Estimation.matfile_read import *
import numpy as np
from numba import njit
import math
import os
from config import gen_sim_data


class TDoA:

    def __init__(self, a_m=2, f_m1=40000000, f_m2=25000000, measure_dt=1e-8, vehicle_dt=1e-3, car_dist=1.6, c=3e8):
        self.a_m = a_m
        self.dt = measure_dt
        self.measure_period = vehicle_dt
        self.w1 = 2 * math.pi * f_m1
        self.w2 = 2 * math.pi * f_m2
        self.car_dist = car_dist
        self.t = np.arange(0, vehicle_dt - self.dt, self.dt)
        self.c = c

    def estimate(self, delays, H, noise_variance):
        delay1_measured, delay2_measured = self.measure_delay(delays, H, noise_variance)
        v = self.c
        ddist1 = np.mean(delay1_measured) * v
        ddist2 = np.mean(delay2_measured) * v
        return np.array([ddist1, ddist2], np.newaxis)

    def measure_delay(self, delays, H, noise_variance):
        # after going through ADC at receiver
        delta_delay1 = delays[0][0] - delays[0][1]
        delta_delay2 = delays[1][0] - delays[1][1]

        s1_w1 = H[0][0] * self.a_m * np.cos(self.w1 * (self.t - delta_delay1)) + np.random.normal(0, math.sqrt(
            noise_variance[0][0]), len(self.t))
        s2_w1 = H[0][1] * self.a_m * np.cos(self.w1 * (self.t)) + np.random.normal(0, math.sqrt(noise_variance[0][1]),
                                                                                   len(self.t))

        s1_w2 = H[1][0] * self.a_m * np.cos(self.w2 * (self.t - delta_delay2)) + np.random.normal(0, math.sqrt(
            noise_variance[1][0]), len(self.t))
        s2_w2 = H[1][1] * self.a_m * np.cos(self.w2 * (self.t)) + np.random.normal(0, math.sqrt(noise_variance[1][1]),
                                                                                   len(self.t))

        s1_w1_fft = np.fft.fft(s1_w1)
        s2_w1_fft = np.fft.fft(s2_w1)

        s1_w1_fft[0:len(s1_w1_fft) // 2] = 0
        s2_w1_fft[0:len(s2_w1_fft) // 2] = 0
        s1_w1_upperSideband = np.fft.ifft(s1_w1_fft)
        s2_w1_upperSideband = np.fft.ifft(s2_w1_fft)

        s1_w2_fft = np.fft.fft(s1_w2)
        s2_w2_fft = np.fft.fft(s2_w2)
        s1_w2_fft[0:len(s1_w2_fft) // 2] = 0
        s2_w2_fft[0:len(s2_w2_fft) // 2] = 0
        s1_w2_upperSideband = np.fft.ifft(s1_w2_fft)
        s2_w2_upperSideband = np.fft.ifft(s2_w2_fft)

        direct_mix1 = np.multiply(s1_w1_upperSideband, s2_w1_upperSideband.conj())
        delay1_measured = np.angle(direct_mix1) / self.w1

        direct_mix2 = np.multiply(s1_w2_upperSideband, s2_w2_upperSideband.conj())
        delay2_measured = np.angle(direct_mix2) / self.w2

        return delay1_measured, delay2_measured


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
        length_time = np.size(t)
        noise1 = np.random.normal(0, math.sqrt(noise_variance[0]), length_time).astype('float')
        noise2 = np.random.normal(0, math.sqrt(noise_variance[1]), length_time).astype('float')
        s_e = np.asarray(signal.square(2 * np.pi * f * t), dtype='float')  # + (noise1 + noise2) / 2
        s_r = np.zeros((2, length_time))
        s_r[0] = np.asarray(signal.square(2 * np.pi * f * (t + delays[0])), dtype='float') + noise1
        s_r[1] = np.asarray(signal.square(2 * np.pi * f * (t + delays[1])), dtype='float') + noise2
        s_h = np.asarray(signal.square(2 * np.pi * f * (r / (r + 1)) * t), dtype='float')
        s_gate = np.asarray((signal.square(2 * np.pi * (f / (N * (r + 1))) * t) > 0), dtype='float')
        return s_e, s_r, s_h, s_gate

    @staticmethod
    @njit(parallel=True)
    def rtof_estimate_dist(s_e, s_r, s_h, s_gate, f, r, N, dt, t, length_time):
        s_clk = np.zeros(length_time);
        s_clk_idx = np.arange(1, length_time, 2);
        s_clk[s_clk_idx] = 1;
        s_phi_hh = np.zeros((2, length_time));
        s_eh_state = 0
        s_rh_state = np.zeros((2))
        counts1 = []
        counts2 = []
        M = np.zeros((2))
        s_h_diff = np.diff(s_h)
        for i in range(1, length_time):

            if s_h_diff[i - 1] == 2:

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

            s_phi_hh[0][i] = np.logical_xor(s_eh_state, s_rh_state[0]) * s_gate[i] * s_clk[i]
            s_phi_hh[1][i] = np.logical_xor(s_eh_state, s_rh_state[1]) * s_gate[i] * s_clk[i]

            if s_gate[i] == 1:
                if s_phi_hh[0][i] == 1:
                    M[0] += 1
                if s_phi_hh[1][i] == 1:
                    M[1] += 1
                update_flag = 1
            else:
                if update_flag == 1:
                    counts1.append(M[0])
                    counts2.append(M[1])
                    M[0] = 0
                    M[1] = 0
                    update_flag = 0
        return counts1, counts2

    def dist_to_pos(self, dm, delays):
        l = self.car_dist
        d1 = dm[0]
        d1_err = np.abs(self.c * delays[1] / 2 - d1)
        d1 = d1[d1_err == np.min(d1_err)][0]
        d2 = dm[1]

        d2_err = np.abs(self.c * delays[0] / 2 - d2)
        d2 = d2[d2_err == np.min(d2_err)][0]
        y = (d2 ** 2 - d1 ** 2 + l ** 2) / (2 * l)
        x = -np.sqrt(d2 ** 2 - y ** 2)

        return d1, d2

    def estimate(self, all_delays, H, noise_variance):

        delay1 = all_delays[0][0] * 2
        delay2 = all_delays[0][1] * 2

        delays = [delay1, delay2]
        s_e, s_r, s_h, s_gate = self.gen_signals(self.f, self.r, self.N, self.t, delays, noise_variance[0])

        s_r[0] *= H[0][0]
        s_r[1] *= H[0][1]

        ### BS: numba and the LLVM optimizer can't really handle varying size arrays well
        ###     so we need to tell the size of the time array beforehand
        ###     also, moved dm computation outside
        length_time = np.size(self.t)
        fclk = 1 / (2 * self.dt)

        counts1, counts2 = self.rtof_estimate_dist(s_e, s_r, s_h, s_gate, self.f, self.r, self.N, self.dt, self.t,
                                                   length_time)
        size_tmp = np.size(counts1)  # could equivalently be counts2
        dm = np.zeros((2, size_tmp));
        dm[0] = ((self.c / 2) * (np.asarray(counts2) / ((self.r + 1) * self.N * fclk)))
        dm[1] = ((self.c / 2) * (np.asarray(counts1) / ((self.r + 1) * self.N * fclk)))

        d11, d12 = self.dist_to_pos(dm, delays)

        delay1 = all_delays[1][0] * 2
        delay2 = all_delays[1][1] * 2

        delays = [delay1, delay2]
        s_e, s_r, s_h, s_gate = self.gen_signals(self.f, self.r, self.N, self.t, delays, noise_variance[1])

        # channel attenuation
        s_r[0] *= H[1][0]
        s_r[1] *= H[1][1]

        counts1, counts2 = self.rtof_estimate_dist(s_e, s_r, s_h, s_gate, self.f, self.r, self.N, self.dt, self.t,
                                                   length_time)
        size_tmp = np.size(counts1)  # could equivalently be counts2
        dm = np.zeros((2, size_tmp));
        dm[0] = ((self.c / 2) * (np.asarray(counts2) / ((self.r + 1) * self.N * fclk)))
        dm[1] = ((self.c / 2) * (np.asarray(counts1) / ((self.r + 1) * self.N * fclk)))

        d21, d22 = self.dist_to_pos(dm, delays)

        dist = np.array([[d11, d12], [d21, d22]])  # to obtain separate axes
        return dist


directory_path = os.path.dirname(os.path.dirname(os.path.join(os.getcwd(), os.listdir(os.getcwd())[0]))) ## directory of directory of file
data = load_mat(directory_path + '/VLP_methods/aoa_transfer_function.mat')
rec_func(data, 0)
breaks = np.array(data['transfer_function']['breaks'])
coefficients = np.array(data['transfer_function']['coefs'])


class AoA:
    def __init__(self, a_m=2, f_m1=1000000, f_m2=2000000, measure_dt=5e-6, vehicle_dt=1e-2, w0=500, hbuf=1000,
                 car_dist=1.6, fov=80):
        self.dt = measure_dt
        self.t = np.arange(0, vehicle_dt - self.dt, self.dt)
        self.w1 = 2 * math.pi * f_m1
        self.w2 = 2 * math.pi * f_m2
        self.a_m = a_m
        self.w0 = w0
        self.hbuf = hbuf
        self.car_dist = car_dist
        self.e_angle = fov

    # %%

    def estimate(self, delays, H_q, noise_variance):
        delta_delay1 = delays[0][0] - delays[0][1]
        delta_delay2 = delays[1][0] - delays[1][1]
        s1_w1 = self.a_m * np.cos(self.w1 * self.t)
        # s1_w1 = s1_w1[:, np.newaxis]
        s2_w2 = self.a_m * np.cos(self.w2 * self.t)
        # s2_w2 = s2_w2[:, np.newaxis]

        # after going through ADC at receiver
        r1_w1_a = H_q[0][0][0] * np.cos(self.w1 * (self.t - delays[0][0])) + np.random.normal(0, math.sqrt(
            noise_variance[0][0][0]), len(self.t))
        r1_w1_b = H_q[0][0][1] * np.cos(self.w1 * (self.t - delays[0][0])) + np.random.normal(0, math.sqrt(
            noise_variance[0][0][1]), len(self.t))
        r1_w1_c = H_q[0][0][2] * np.cos(self.w1 * (self.t - delays[0][0])) + np.random.normal(0, math.sqrt(
            noise_variance[0][0][2]), len(self.t))
        r1_w1_d = H_q[0][0][3] * np.cos(self.w1 * (self.t - delays[0][0])) + np.random.normal(0, math.sqrt(
            noise_variance[0][0][3]), len(self.t))

        r2_w1_a = H_q[0][1][0] * np.cos(self.w1 * (self.t - delays[0][1])) + np.random.normal(0, math.sqrt(
            noise_variance[0][1][0]), len(self.t))
        r2_w1_b = H_q[0][1][1] * np.cos(self.w1 * (self.t - delays[0][1])) + np.random.normal(0, math.sqrt(
            noise_variance[0][1][1]), len(self.t))
        r2_w1_c = H_q[0][1][2] * np.cos(self.w1 * (self.t - delays[0][1])) + np.random.normal(0, math.sqrt(
            noise_variance[0][1][2]), len(self.t))
        r2_w1_d = H_q[0][1][3] * np.cos(self.w1 * (self.t - delays[0][1])) + np.random.normal(0, math.sqrt(
            noise_variance[0][1][3]), len(self.t))

        r1_w2_a = H_q[1][0][0] * np.cos(self.w2 * (self.t - delays[1][0])) + np.random.normal(0, math.sqrt(
            noise_variance[1][0][0]), len(self.t))
        r1_w2_b = H_q[1][0][1] * np.cos(self.w2 * (self.t - delays[1][0])) + np.random.normal(0, math.sqrt(
            noise_variance[1][0][1]), len(self.t))
        r1_w2_c = H_q[1][0][2] * np.cos(self.w2 * (self.t - delays[1][0])) + np.random.normal(0, math.sqrt(
            noise_variance[1][0][2]), len(self.t))
        r1_w2_d = H_q[1][0][3] * np.cos(self.w2 * (self.t - delays[1][0])) + np.random.normal(0, math.sqrt(
            noise_variance[1][0][3]), len(self.t))

        r2_w2_a = H_q[1][1][0] * np.cos(self.w2 * (self.t - delays[1][1])) + np.random.normal(0, math.sqrt(
            noise_variance[1][1][0]), len(self.t))
        r2_w2_b = H_q[1][1][1] * np.cos(self.w2 * (self.t - delays[1][1])) + np.random.normal(0, math.sqrt(
            noise_variance[1][1][1]), len(self.t))
        r2_w2_c = H_q[1][1][2] * np.cos(self.w2 * (self.t - delays[1][1])) + np.random.normal(0, math.sqrt(
            noise_variance[1][1][2]), len(self.t))
        r2_w2_d = H_q[1][1][3] * np.cos(self.w2 * (self.t - delays[1][1])) + np.random.normal(0, math.sqrt(
            noise_variance[1][1][3]), len(self.t))

        eps_a_s1, eps_b_s1, eps_c_s1, eps_d_s1, phi_h_s1 = np.array([0., 0.]), np.array(
            [0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])
        eps_a_s2, eps_b_s2, eps_c_s2, eps_d_s2, phi_h_s2 = np.array([0., 0.]), np.array(
            [0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])
        theta_l_r = np.array([[0., 0.], [0., 0.]]).astype(float)

        eps_a_s1[0] = np.sum(
            np.dot(r1_w1_a[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_b_s1[0] = np.sum(
            np.dot(r1_w1_b[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_c_s1[0] = np.sum(
            np.dot(r1_w1_c[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_d_s1[0] = np.sum(
            np.dot(r1_w1_d[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_a_s1[1] = np.sum(
            np.dot(r2_w1_a[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_b_s1[1] = np.sum(
            np.dot(r2_w1_b[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_c_s1[1] = np.sum(
            np.dot(r2_w1_c[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_d_s1[1] = np.sum(
            np.dot(r2_w1_d[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf

        eps_a_s2[0] = np.sum(
            np.dot(r1_w2_a[self.w0: self.w0 + self.hbuf], s2_w2[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_b_s2[0] = np.sum(
            np.dot(r1_w2_b[self.w0: self.w0 + self.hbuf], s2_w2[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_c_s2[0] = np.sum(
            np.dot(r1_w2_c[self.w0: self.w0 + self.hbuf], s2_w2[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_d_s2[0] = np.sum(
            np.dot(r1_w2_d[self.w0: self.w0 + self.hbuf], s2_w2[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_a_s2[1] = np.sum(
            np.dot(r2_w2_a[self.w0: self.w0 + self.hbuf], s2_w2[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_b_s2[1] = np.sum(
            np.dot(r2_w2_b[self.w0: self.w0 + self.hbuf], s2_w2[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_c_s2[1] = np.sum(
            np.dot(r2_w2_c[self.w0: self.w0 + self.hbuf], s2_w2[self.w0: self.w0 + self.hbuf])) / self.hbuf
        eps_d_s2[1] = np.sum(
            np.dot(r2_w2_d[self.w0: self.w0 + self.hbuf], s2_w2[self.w0: self.w0 + self.hbuf])) / self.hbuf


        phi_h_s1[0] = ((eps_b_s1[0] + eps_d_s1[0]) - (eps_a_s1[0] + eps_c_s1[0])) / (
                eps_a_s1[0] + eps_b_s1[0] + eps_c_s1[0] + eps_d_s1[0])
        phi_h_s1[1] = ((eps_b_s1[1] + eps_d_s1[1]) - (eps_a_s1[1] + eps_c_s1[1])) / (
                eps_a_s1[1] + eps_b_s1[1] + eps_c_s1[1] + eps_d_s1[1])
        phi_h_s2[0] = ((eps_b_s2[0] + eps_d_s2[0]) - (eps_a_s2[0] + eps_c_s2[0])) / (
                eps_a_s2[0] + eps_b_s2[0] + eps_c_s2[0] + eps_d_s2[0])
        phi_h_s2[1] = ((eps_b_s2[1] + eps_d_s2[1]) - (eps_a_s2[1] + eps_c_s2[1])) / (
                eps_a_s2[1] + eps_b_s2[1] + eps_c_s2[1] + eps_d_s2[1])

        theta_l_r[0][0] = self.transfer_function(phi_h_s1[0]) * np.pi / 180
        theta_l_r[0][1] = self.transfer_function(phi_h_s1[1]) * np.pi / 180
        theta_l_r[1][0] = self.transfer_function(phi_h_s2[0]) * np.pi / 180
        theta_l_r[1][1] = self.transfer_function(phi_h_s2[1]) * np.pi / 180
        return theta_l_r

    # %%

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def transfer(self, coefficient, x, x1):
        return (coefficient[0] * (x - x1) ** 3) + (coefficient[1] * (x - x1) ** 2) + (coefficient[2] * (x - x1) ** 1) + \
               coefficient[3]

    def transfer_function(self, phi):
        phi = 1.0000653324773283 if phi >= 1.0000653324773283 else phi
        phi = -1.0000980562352184 if phi <= -1.0000980562352184 else phi

        idx, lower_endpoint = self.find_nearest(breaks, phi)
        coefficient = coefficients[idx]
        return self.transfer(coefficient, phi, lower_endpoint)

    def change_cords(self, txpos):
        t_tx_pos = np.copy(txpos)
        t_tx_pos[0][0] = -txpos[0][1]
        t_tx_pos[1][0] = txpos[0][0]
        t_tx_pos[0][1] = -txpos[1][1]
        t_tx_pos[1][1] = txpos[1][0]
        return t_tx_pos


def main():
    data_names = gen_sim_data.names.data_names
    folder_names = gen_sim_data.names.folder_names
    size = gen_sim_data.params.end_point_of_iter - gen_sim_data.params.start_point_of_iter

    theta_l_r = np.zeros((size, 100, 2, 2))
    rtof_dist = np.zeros((size, 100, 2, 2))
    tdoa_dist = np.zeros((size, 100, 2))

    # run for multiple iterations

    for itr in range(gen_sim_data.params.start_point_of_iter, gen_sim_data.params.end_point_of_iter):
        print(itr)
        for idx in range(2, 3):
            data_name = data_names[idx]
            data_dir = directory_path + '/SimulationData/' + data_name + '.mat'
            data = load_mat(data_dir)
            folder_name = folder_names[idx]
            dp = gen_sim_data.params.number_of_skip_data
            data_point = str(int(1000 / dp)) + '_point_' + '/'

            f_name = directory_path + '/Parameter_Deviation/' + data_point + folder_name

            if not os.path.exists(f_name):
                os.makedirs(f_name)

            max_power = data['tx']['power']
            area = data['qrx']['f_QRX']['params']['area']
            rx_radius = math.sqrt(area) / math.pi

            c = gen_sim_data.params.c
            rx_fov = gen_sim_data.params.rx_fov
            tx_half_angle = gen_sim_data.params.tx_half_angle
            signal_freq = gen_sim_data.params.signal_freq
            measure_dt = gen_sim_data.params.measure_dt

            time_ = data['vehicle']['t']['values']
            time_ = time_[::dp]
            vehicle_dt = data['vehicle']['t']['dt']

            rel_hdg = data['vehicle']['target_relative']['heading'][::dp]

            L_tgt = data['vehicle']['target']['width']
            L_ego = data['vehicle']['ego']['width']

            tgt_tx1_x = -1 * data['vehicle']['target_relative']['tx1_qrx4']['y'][::dp]
            tgt_tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x'][::dp]
            tgt_tx2_x = -1 * data['vehicle']['target_relative']['tx2_qrx3']['y'][::dp]
            tgt_tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x'][::dp]

            # delay parameters
            delay_11 = data['channel']['qrx1']['delay']['tx1'][::dp]
            delay_12 = data['channel']['qrx1']['delay']['tx2'][::dp]
            delay_21 = data['channel']['qrx2']['delay']['tx1'][::dp]
            delay_22 = data['channel']['qrx2']['delay']['tx2'][::dp]

            # received power of QRXes
            pow_qrx1_tx1 = np.array(
                [data['channel']['qrx1']['power']['tx1']['A'][::dp], data['channel']['qrx1']['power']['tx1']['B'][::dp],
                 data['channel']['qrx1']['power']['tx1']['C'][::dp],
                 data['channel']['qrx1']['power']['tx1']['D'][::dp]])
            pow_qrx1_tx2 = np.array(
                [data['channel']['qrx1']['power']['tx2']['A'][::dp], data['channel']['qrx1']['power']['tx2']['B'][::dp],
                 data['channel']['qrx1']['power']['tx2']['C'][::dp],
                 data['channel']['qrx1']['power']['tx2']['D'][::dp]])
            pow_qrx2_tx1 = np.array(
                [data['channel']['qrx2']['power']['tx1']['A'][::dp], data['channel']['qrx2']['power']['tx1']['B'][::dp],
                 data['channel']['qrx2']['power']['tx1']['C'][::dp],
                 data['channel']['qrx2']['power']['tx1']['D'][::dp]])
            pow_qrx2_tx2 = np.array(
                [data['channel']['qrx2']['power']['tx1']['A'][::dp], data['channel']['qrx2']['power']['tx1']['B'][::dp],
                 data['channel']['qrx2']['power']['tx1']['C'][::dp],
                 data['channel']['qrx2']['power']['tx1']['D'][::dp]])

            # noise params
            T = gen_sim_data.params.T
            I_bg = gen_sim_data.params.I_bg
            p_r_factor = data['qrx']['tia']['shot_P_r_factor']
            i_bg_factor = data['qrx']['tia']['shot_I_bg_factor']
            t_factor1 = data['qrx']['tia']['thermal_factor1']
            t_factor2 = data['qrx']['tia']['thermal_factor1']

            x, y, x_pose, y_pose, x_roberts, y_roberts, x_becha, y_becha = np.zeros((len(tgt_tx1_x), 2)), np.zeros(
                (len(tgt_tx1_x),
                 2)), \
                                                                           np.zeros((len(tgt_tx1_x), 2)), np.zeros(
                (len(tgt_tx1_x),
                 2)), \
                                                                           np.zeros((len(tgt_tx1_x), 2)), np.zeros(
                (len(tgt_tx1_x),
                 2)), \
                                                                           np.zeros((len(tgt_tx1_x), 2)), np.zeros(
                (len(tgt_tx1_x),
                 2))


            aoa = AoA(a_m=max_power, f_m1=signal_freq, f_m2=2 * signal_freq, measure_dt=measure_dt,
                      vehicle_dt=vehicle_dt * dp,
                      w0=gen_sim_data.params.w0, hbuf=int(vehicle_dt / measure_dt), car_dist=L_tgt, fov=rx_fov)
            rtof = RToF(a_m=max_power, f_m=signal_freq, measure_dt=gen_sim_data.params.rtof_measure_dt,
                        vehicle_dt=vehicle_dt * dp, car_dist=L_tgt,
                        r=gen_sim_data.params.r, N=gen_sim_data.params.N, c=c)
            tdoa = TDoA(a_m=max_power, f_m1=signal_freq, f_m2=signal_freq, measure_dt=measure_dt,
                        vehicle_dt=vehicle_dt * dp,
                        car_dist=L_tgt)
            for i in range(len(tgt_tx1_x)):
                # updating the given coordinates
                print("Iteration #", i, ": ")
                x[i] = (tgt_tx1_x[i], tgt_tx2_x[i])
                y[i] = (tgt_tx1_y[i], tgt_tx2_y[i])

                # providing the environment to methods
                delays = np.array([[delay_11[i], delay_21[i]], [delay_12[i], delay_22[i]]])
                H_q = np.array([[pow_qrx1_tx1[:, i], pow_qrx2_tx1[:, i]], [pow_qrx1_tx2[:, i], pow_qrx2_tx2[:, i]]])
                H = np.array([[np.sum(pow_qrx1_tx1[:, i]), np.sum(pow_qrx2_tx1[:, i])],
                              [np.sum(pow_qrx1_tx2[:, i]), np.sum(pow_qrx2_tx2[:, i])]])

                p_r1, p_r2, p_r3, p_r4 = H[0][0], H[0][1], H[1][0], H[1][1]
                remaining_factor = I_bg * i_bg_factor + T * (t_factor1 + t_factor2)
                noise_var1 = p_r1 * p_r_factor + remaining_factor
                noise_var2 = p_r2 * p_r_factor + remaining_factor
                noise_var3 = p_r3 * p_r_factor + remaining_factor
                noise_var4 = p_r4 * p_r_factor + remaining_factor
                noise_variance = np.array([[noise_var1, noise_var2], [noise_var3, noise_var4]])

                # noise_variance = np.array([[0.0, 0.0], [0.0, 0.0]])
                rem_fact_soner = I_bg * i_bg_factor + T * (t_factor1 + t_factor2 / 16)
                noise_var1_soner = np.array(
                    [H_q[0][0][0] * p_r_factor + rem_fact_soner, H_q[0][0][1] * p_r_factor + rem_fact_soner,
                     H_q[0][0][2] * p_r_factor + rem_fact_soner, H_q[0][0][3] * p_r_factor + rem_fact_soner])
                noise_var2_soner = np.array(
                    [H_q[0][1][0] * p_r_factor + rem_fact_soner, H_q[0][1][1] * p_r_factor + rem_fact_soner,
                     H_q[0][1][2] * p_r_factor + rem_fact_soner, H_q[0][1][3] * p_r_factor + rem_fact_soner])
                noise_var3_soner = np.array(
                    [H_q[1][0][0] * p_r_factor + rem_fact_soner, H_q[1][0][1] * p_r_factor + rem_fact_soner,
                     H_q[1][0][2] * p_r_factor + rem_fact_soner, H_q[1][0][3] * p_r_factor + rem_fact_soner])
                noise_var4_soner = np.array(
                    [H_q[1][1][0] * p_r_factor + rem_fact_soner, H_q[1][1][1] * p_r_factor + rem_fact_soner,
                     H_q[1][1][2] * p_r_factor + rem_fact_soner, H_q[1][1][3] * p_r_factor + rem_fact_soner])
                noise_variance_soner = np.array(
                    [[noise_var1_soner, noise_var2_soner], [noise_var3_soner, noise_var4_soner]])

                # making estimations
                #theta_l_r[itr, i] = aoa.estimate(delays=delays, H_q=H_q, noise_variance=noise_variance_soner)
                #print("AoA finished")
                rtof_dist[itr, i] = rtof.estimate(all_delays=delays, H=H, noise_variance=noise_variance)
                print("RToF finished")
                #tdoa_dist[itr, i] = tdoa.estimate(delays=delays, H=H, noise_variance=noise_variance)
                #print("TDoA finished")
                #print(time_)
    pickle_dir = directory_path + '/Bound_Estimation/Parameter_Deviation/'
    #with open(pickle_dir + 'theta.pkl', 'wb') as f:
    #    pickle.dump(theta_l_r, f)
    with open(pickle_dir + 'rtof_dist', 'wb') as f:
        pickle.dump(rtof_dist, f)
    #with open(pickle_dir + 'tdoa_dist.pkl', 'wb') as f:
    #   pickle.dump(tdoa_dist, f)


if __name__ == '__main__':
    main()
