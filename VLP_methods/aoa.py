# %%

from cache.VLC_init import *
from Bound_Estimation.matfile_read import *
import math
from scipy.io import loadmat, matlab

# %%

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

data = load_mat('VLP_methods/aoa_transfer_function.mat')
rec_func(data, 0)
breaks = np.array(data['transfer_function']['breaks'])
coefficients = np.array(data['transfer_function']['coefs'])


# %%

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

        # print(self.t.shape)
        # print(self.dt)
        # print(r1_w1_a.shape)
        # print(s1_w1.shape)
        # print(np.dot(r1_w1_a[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf]).shape)
        # print(np.dot(r1_w1_a[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf]))
        # print(np.sum(np.dot(r1_w1_a[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])))
        # print(np.sum(np.dot(r1_w1_a[self.w0: self.w0 + self.hbuf], s1_w1[self.w0: self.w0 + self.hbuf])) / self.hbuf)

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

        # print(eps_a_s1)
        # print(eps_a_s2)
        # print(eps_b_s1)
        # print(eps_b_s2)
        # print(eps_c_s1)
        # print(eps_c_s2)
        # print(eps_d_s1)
        # print(eps_d_s2)

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

        #print(theta_l_r)
        # %%

        diff_1 = theta_l_r[0][0] - theta_l_r[0][1]
        t_x_1 = self.car_dist * (1 + (math.sin(theta_l_r[0][1]) * math.cos(theta_l_r[0][0])) / (math.sin(diff_1))) if math.sin(diff_1) != 0 else None
        t_y_1 = self.car_dist * ((math.cos(theta_l_r[0][1]) * math.cos(theta_l_r[0][0])) / (math.sin(diff_1))) if math.sin(diff_1) != 0 else None
        # print("Transmitter-1 x pos is : ", self.t_x_1, ", y pos is : ", self.t_y_1)

        diff_2 = theta_l_r[1][0] - theta_l_r[1][1]
        t_x_2 = self.car_dist * (1 + (math.sin(theta_l_r[1][1]) * math.cos(theta_l_r[1][0])) / (math.sin(diff_2))) if math.sin(diff_2) != 0 else None
        t_y_2 = self.car_dist * ((math.cos(theta_l_r[1][1]) * math.cos(theta_l_r[1][0])) / (math.sin(diff_2))) if math.sin(diff_2) != 0 else None
        # print(t_x_1, t_x_2, t_y_1, t_y_2)
        # exit(0)
        # print("Transmitter-2 x pos is : ", self.t_x_2, ", y pos is : ", self.t_y_2)
        # %%
        # Error calc
        # print("Error of Transmitter-1 position in x:", abs(self.t_x_1_act-self.t_x_1),", y:", abs(self.t_y_1_act-self.t_y_1))
        # print("Error of Transmitter-2 position in x, y:", abs(self.t_x_2_act-self.t_x_2),", y:", abs(self.t_y_2_act-self.t_y_2))
        tx_pos = self.change_cords(np.array([[t_x_1, t_y_1], [t_x_2, t_y_2]]))
        # print("Transmitter-1 x pos is: ", tx_pos[0][0], ", y pos is : ", tx_pos[0][1])
        # print("Transmitter-2 x pos is : ", tx_pos[1][0], ", y pos is : ", tx_pos[1][1])

        return tx_pos

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

# %%
