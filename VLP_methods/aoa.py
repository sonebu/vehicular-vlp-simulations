# %%

from cache.VLC_init import *
import math

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


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)
    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def change_cords(txpos):
    t_tx_pos = np.copy(txpos)
    t_tx_pos[0][0] = txpos[0][1]
    t_tx_pos[1][0] = (-1 * txpos[0][0])
    t_tx_pos[0][1] = txpos[1][1]
    t_tx_pos[1][1] = (-1 * txpos[1][0])
    return t_tx_pos


# %%

class AoA:
    def __init__(self, a_m=2, f_m1=1000000, f_m2=2000000, measure_dt=5e-6, vehicle_dt=1e-2, w0=500, hbuf=1000, car_dist=1.6, fov=80):
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
        s1_w1 = np.cos(self.w1 * self.t)
        #s1_w1 = s1_w1[:, np.newaxis]
        s2_w2 = np.cos(self.w2 * self.t)
        #s2_w2 = s2_w2[:, np.newaxis]

        H = np.array([[np.sum(H_q[0][0]), np.sum(H_q[0][1])], [np.sum(H_q[1][0]), np.sum(H_q[1][1])]])
        # after going through ADC at receiver
        r1_w1 = H[0][0] * np.cos(self.w1 * (self.t - delta_delay1)) + np.random.normal(0, math.sqrt(noise_variance[0][0]), len(self.t))
        r2_w1 = H[1][0] * np.cos(self.w1 * self.t) + np.random.normal(0, math.sqrt(noise_variance[1][0]), len(self.t))

        r1_w2 = H[0][1] * np.cos(self.w2 * (self.t - delta_delay2)) + np.random.normal(0, math.sqrt(noise_variance[0][1]), len(self.t))
        r2_w2 = H[1][1] * np.cos(self.w2 * self.t) + np.random.normal(0, math.sqrt(noise_variance[1][1]), len(self.t))

        eps_a_s1, eps_b_s1, eps_c_s1, eps_d_s1, phi_h_s1 = np.array([0., 0.]), np.array(
            [0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])
        eps_a_s2, eps_b_s2, eps_c_s2, eps_d_s2, phi_h_s2 = np.array([0., 0.]), np.array(
            [0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])
        theta_R_L = np.array([[0., 0.], [0., 0.]]).astype(float)
        for w in range(self.w0, (self.w0 + self.hbuf - 1)):
            # H_q * np.sum(np.dot(,)) TODO
            eps_a_s1[0] += H_q[0][0][0] * r1_w1[w] * s1_w1[w]
            eps_b_s1[0] += H_q[0][0][1] * r1_w1[w] * s1_w1[w]
            eps_c_s1[0] += H_q[0][0][2] * r1_w1[w] * s1_w1[w]
            eps_d_s1[0] += H_q[0][0][2] * r1_w1[w] * s1_w1[w]
            eps_a_s1[1] += H_q[0][1][0] * r2_w1[w] * s1_w1[w]
            eps_b_s1[1] += H_q[0][1][1] * r2_w1[w] * s1_w1[w]
            eps_c_s1[1] += H_q[0][1][2] * r2_w1[w] * s1_w1[w]
            eps_d_s1[1] += H_q[0][1][3] * r2_w1[w] * s1_w1[w]

            eps_a_s2[0] += H_q[1][0][0] * r1_w2[w] * s2_w2[w]
            eps_b_s2[0] += H_q[1][0][0] * r1_w2[w] * s2_w2[w]
            eps_c_s2[0] += H_q[1][0][0] * r1_w2[w] * s2_w2[w]
            eps_d_s2[0] += H_q[1][0][0] * r1_w2[w] * s2_w2[w]
            eps_a_s2[1] += H_q[1][1][0] * r2_w2[w] * s2_w2[w]
            eps_b_s2[1] += H_q[1][1][1] * r2_w2[w] * s2_w2[w]
            eps_c_s2[1] += H_q[1][1][2] * r2_w2[w] * s2_w2[w]
            eps_d_s2[1] += H_q[1][1][3] * r2_w2[w] * s2_w2[w]

        eps_a_s1[0] /= self.hbuf
        eps_b_s1[0] /= self.hbuf
        eps_c_s1[0] /= self.hbuf
        eps_d_s1[0] /= self.hbuf
        eps_a_s1[1] /= self.hbuf
        eps_b_s1[1] /= self.hbuf
        eps_c_s1[1] /= self.hbuf
        eps_d_s1[1] /= self.hbuf

        eps_a_s2[0] /= self.hbuf
        eps_b_s2[0] /= self.hbuf
        eps_c_s2[0] /= self.hbuf
        eps_d_s2[0] /= self.hbuf
        eps_a_s2[1] /= self.hbuf
        eps_b_s2[1] /= self.hbuf
        eps_c_s2[1] /= self.hbuf
        eps_d_s2[1] /= self.hbuf

        phi_h_s1[0] = ((eps_b_s1[0] + eps_d_s1[0]) - (eps_a_s1[0] + eps_c_s1[0])) / (
                eps_a_s1[0] + eps_b_s1[0] + eps_c_s1[0] + eps_d_s1[0])
        phi_h_s1[1] = ((eps_b_s1[1] + eps_d_s1[1]) - (eps_a_s1[1] + eps_c_s1[1])) / (
                eps_a_s1[1] + eps_b_s1[1] + eps_c_s1[1] + eps_d_s1[1])
        phi_h_s2[0] = ((eps_b_s2[0] + eps_d_s2[0]) - (eps_a_s2[0] + eps_c_s2[0])) / (
                eps_a_s2[0] + eps_b_s2[0] + eps_c_s2[0] + eps_d_s2[0])
        phi_h_s2[1] = ((eps_b_s2[1] + eps_d_s2[1]) - (eps_a_s2[1] + eps_c_s2[1])) / (
                eps_a_s2[1] + eps_b_s2[1] + eps_c_s2[1] + eps_d_s2[1])

        theta_R_L[0][0] = translate(phi_h_s1[0], -1, 1, (-1 * self.e_angle), self.e_angle)
        theta_R_L[0][1] = translate(phi_h_s1[1], -1, 1, (-1 * self.e_angle), self.e_angle)
        theta_R_L[1][0] = translate(phi_h_s2[0], -1, 1, (-1 * self.e_angle), self.e_angle)
        theta_R_L[1][1] = translate(phi_h_s2[1], -1, 1, (-1 * self.e_angle), self.e_angle)

        # %%

        diff_1 = theta_R_L[0][1] - theta_R_L[0][0]
        t_x_1 = self.car_dist * ((self.car_dist/2) + (math.sin(theta_R_L[0][0]) * math.cos(theta_R_L[0][1])) / (math.sin(diff_1)))
        t_y_1 = self.car_dist * ( (math.cos(theta_R_L[0][0]) * math.cos(theta_R_L[0][1])) / (math.sin(diff_1)))
        # print("Transmitter-1 x pos is : ", self.t_x_1, ", y pos is : ", self.t_y_1)

        diff_2 = theta_R_L[1][1] - theta_R_L[1][0]
        t_x_2 = self.car_dist * ((self.car_dist/2) + (math.sin(theta_R_L[1][0]) * math.cos(theta_R_L[1][1])) / (math.sin(diff_2)))
        t_y_2 = self.car_dist * ((math.cos(theta_R_L[1][0]) * math.cos(theta_R_L[1][1])) / (math.sin(diff_2)))
        # print("Transmitter-2 x pos is : ", self.t_x_2, ", y pos is : ", self.t_y_2)
        # %%
        # Error calc
        # print("Error of Transmitter-1 position in x:", abs(self.t_x_1_act-self.t_x_1),", y:", abs(self.t_y_1_act-self.t_y_1))
        # print("Error of Transmitter-2 position in x, y:", abs(self.t_x_2_act-self.t_x_2),", y:", abs(self.t_y_2_act-self.t_y_2))
        tx_pos = change_cords(np.array([[t_x_1, t_y_1], [t_x_2, t_y_2]]))
        #print("Transmitter-1 x pos is: ", tx_pos[0][0], ", y pos is : ", tx_pos[0][1])
        #print("Transmitter-2 x pos is : ", tx_pos[1][0], ", y pos is : ", tx_pos[1][1])

        return tx_pos
# %%


# %%