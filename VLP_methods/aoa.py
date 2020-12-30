# %%

from VLP_methods.VLC_init import *
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

class Pose:
    def __init__(self, vlcobj):
        self.f_m1 = 1000000  # 1MHz
        self.f_m2 = 2000000  # 2MHz
        self.dt = 5e-6
        self.t = np.arange(0, 1e-2 - self.dt, self.dt)
        self.w1 = 2 * math.pi * self.f_m1
        self.w2 = 2 * math.pi * self.f_m2
        self.noise_standard_deviation = 1e-5
        self.vlc_obj = vlcobj
        self.t_x_1_act, self.t_y_1_act, self.t_x_2_act, self.t_y_2_act = self.vlc_obj.change_cords()
        self.w0 = 500
        self.hbuf = 1000
        self.u_q = 1e-12

    # %%

    def estimate(self):
        delta_delay1 = self.vlc_obj.delays[0][0] - self.vlc_obj.delays[0][1]
        delta_delay2 = self.vlc_obj.delays[1][0] - self.vlc_obj.delays[1][1]
        s1_w1 = np.cos(self.w1 * self.t)
        s1_w1 = s1_w1[:, np.newaxis].T
        s2_w2 = np.cos(self.w2 * self.t)
        s2_w2 = s2_w2[:, np.newaxis].T

        # after going through ADC at receiver
        r1_w1 = (self.vlc_obj.H[0][0] * np.cos(self.w1 * (self.t - delta_delay1)) + self.noise_standard_deviation * np.random.randn(1, len(self.t)))
        r2_w1 = (self.vlc_obj.H[1][0] * np.cos(self.w1 * self.t) + self.noise_standard_deviation * np.random.randn(1, len(self.t)))

        r1_w2 = (self.vlc_obj.H[0][1] * np.cos(self.w2 * (self.t - delta_delay2)) + self.noise_standard_deviation * np.random.randn(1, len(self.t)))
        r2_w2 = (self.vlc_obj.H[1][1] * np.cos(self.w2 * self.t) + self.noise_standard_deviation * np.random.randn(1, len(self.t)))
        eps_a_s1, eps_b_s1, eps_c_s1, eps_d_s1, phi_h_s1 = np.array([0., 0.]), np.array(
            [0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])
        eps_a_s2, eps_b_s2, eps_c_s2, eps_d_s2, phi_h_s2 = np.array([0., 0.]), np.array(
            [0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])
        theta_R_L = np.array([[0., 0.], [0., 0.]]).astype(float)
        for w in range(self.w0, (self.w0 + self.hbuf - 1)):
            eps_a_s1[0] += ((self.vlc_obj.eps_a[0][0] * r1_w1[0][w] - self.u_q) * s1_w1[0][w])
            eps_b_s1[0] += ((self.vlc_obj.eps_b[0][0] * r1_w1[0][w] - self.u_q) * s1_w1[0][w])
            eps_c_s1[0] += ((self.vlc_obj.eps_c[0][0] * r1_w1[0][w] - self.u_q) * s1_w1[0][w])
            eps_d_s1[0] += ((self.vlc_obj.eps_d[0][0] * r1_w1[0][w] - self.u_q) * s1_w1[0][w])
            eps_a_s1[1] += ((self.vlc_obj.eps_a[0][1] * r2_w1[0][w] - self.u_q) * s1_w1[0][w])
            eps_b_s1[1] += ((self.vlc_obj.eps_b[0][1] * r2_w1[0][w] - self.u_q) * s1_w1[0][w])
            eps_c_s1[1] += ((self.vlc_obj.eps_c[0][1] * r2_w1[0][w] - self.u_q) * s1_w1[0][w])
            eps_d_s1[1] += ((self.vlc_obj.eps_d[0][1] * r2_w1[0][w] - self.u_q) * s1_w1[0][w])

            eps_a_s2[0] += ((self.vlc_obj.eps_a[1][0] * r1_w2[0][w] - self.u_q) * s2_w2[0][w])
            eps_b_s2[0] += ((self.vlc_obj.eps_b[1][0] * r1_w2[0][w] - self.u_q) * s2_w2[0][w])
            eps_c_s2[0] += ((self.vlc_obj.eps_c[1][0] * r1_w2[0][w] - self.u_q) * s2_w2[0][w])
            eps_d_s2[0] += ((self.vlc_obj.eps_d[1][0] * r1_w2[0][w] - self.u_q) * s2_w2[0][w])
            eps_a_s2[1] += ((self.vlc_obj.eps_a[1][1] * r2_w2[0][w] - self.u_q) * s2_w2[0][w])
            eps_b_s2[1] += ((self.vlc_obj.eps_b[1][1] * r2_w2[0][w] - self.u_q) * s2_w2[0][w])
            eps_c_s2[1] += ((self.vlc_obj.eps_c[1][1] * r2_w2[0][w] - self.u_q) * s2_w2[0][w])
            eps_d_s2[1] += ((self.vlc_obj.eps_d[1][1] * r2_w2[0][w] - self.u_q) * s2_w2[0][w])

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

        theta_R_L[0][0] = self.vlc_obj.translate(phi_h_s1[0], -1, 1, (-1 * self.vlc_obj.e_angle),
                                                 self.vlc_obj.e_angle)
        theta_R_L[0][1] = self.vlc_obj.translate(phi_h_s1[1], -1, 1, (-1 * self.vlc_obj.e_angle),
                                                 self.vlc_obj.e_angle)
        theta_R_L[1][0] = self.vlc_obj.translate(phi_h_s2[0], -1, 1, (-1 * self.vlc_obj.e_angle),
                                                 self.vlc_obj.e_angle)
        theta_R_L[1][1] = self.vlc_obj.translate(phi_h_s2[1], -1, 1, (-1 * self.vlc_obj.e_angle),
                                                 self.vlc_obj.e_angle)

        # %%

        diff_1 = theta_R_L[0][1] - theta_R_L[0][0]
        t_x_1 = self.vlc_obj.distancecar * (
                (self.vlc_obj.distancecar/2) + (math.sin(theta_R_L[0][0]) * math.cos(theta_R_L[0][1])) / (math.sin(diff_1)))
        t_y_1 = self.vlc_obj.distancecar * (
                (math.cos(theta_R_L[0][0]) * math.cos(theta_R_L[0][1])) / (math.sin(diff_1)))
        # print("Transmitter-1 x pos is : ", self.t_x_1, ", y pos is : ", self.t_y_1)

        diff_2 = theta_R_L[1][1] - theta_R_L[1][0]
        t_x_2 = self.vlc_obj.distancecar * (
                (self.vlc_obj.distancecar/2) + (math.sin(theta_R_L[1][0]) * math.cos(theta_R_L[1][1])) / (math.sin(diff_2)))
        t_y_2 = self.vlc_obj.distancecar * (
                (math.cos(theta_R_L[1][0]) * math.cos(theta_R_L[1][1])) / (math.sin(diff_2)))
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