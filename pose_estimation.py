# %%

from VLC_init import *
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
        self.delta_delay1 = self.vlc_obj.delays[0][0] - self.vlc_obj.delays[0][1]
        self.delta_delay2 = self.vlc_obj.delays[1][0] - self.vlc_obj.delays[1][1]
        self.s1_w1, self.s2_w2 = 0, 0
        # after going through ADC at receiver
        self.r1_w1, self.r2_w1, self.r1_w2, self.r2_w2 = None, None, None, None
        self.eps_a_s1, self.eps_b_s1, self.eps_c_s1, self.eps_d_s1, self.phi_h_s1 = np.array([0., 0.]), np.array(
            [0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])
        self.eps_a_s2, self.eps_b_s2, self.eps_c_s2, self.eps_d_s2, self.phi_h_s2 = np.array([0., 0.]), np.array(
            [0., 0.]), np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.])

        self.hbuf, self.w0 = 1000, 500
        self.theta_R_L = np.array([[0., 0.], [0., 0.]]).astype(float)
        self.u_q = 1e-12

        self.diff_1, self.t_x_1, self.t_y_1, self.diff_2, self.t_x_2, self.t_y_2 = None, None, None, None, None, None
        self.t_x_1_act, self.t_y_1_act, self.t_x_2_act, self.t_y_2_act = self.vlc_obj.change_cords()

    # %%

    def estimate(self):
        self.delta_delay1 = self.vlc_obj.delays[0][0] - self.vlc_obj.delays[0][1]
        self.delta_delay2 = self.vlc_obj.delays[1][0] - self.vlc_obj.delays[1][1]
        self.s1_w1 = np.cos(self.w1 * self.t)
        self.s1_w1 = self.s1_w1[:, np.newaxis].T
        self.s2_w2 = np.cos(self.w2 * self.t)
        self.s2_w2 = self.s2_w2[:, np.newaxis].T

        # after going through ADC at receiver
        self.r1_w1 = (self.vlc_obj.H[0][0] * np.cos(
            self.w1 * (self.t - self.delta_delay1)) + self.noise_standard_deviation * np.random.randn(1, len(self.t)))
        self.r2_w1 = (
                    self.vlc_obj.H[1][0] * np.cos(self.w1 * self.t) + self.noise_standard_deviation * np.random.randn(1,
                                                                                                                      len(
                                                                                                                          self.t)))

        self.r1_w2 = (self.vlc_obj.H[0][1] * np.cos(
            self.w2 * (self.t - self.delta_delay2)) + self.noise_standard_deviation * np.random.randn(1, len(self.t)))
        self.r2_w2 = (
                    self.vlc_obj.H[1][1] * np.cos(self.w2 * self.t) + self.noise_standard_deviation * np.random.randn(1,
                                                                                                                      len(
                                                                                                                          self.t)))

        for w in range(self.w0, (self.w0 + self.hbuf - 1)):
            self.eps_a_s1[0] += ((self.vlc_obj.eps_a[0][0] * self.r1_w1[0][w] - self.u_q) * self.s1_w1[0][w])
            self.eps_b_s1[0] += ((self.vlc_obj.eps_b[0][0] * self.r1_w1[0][w] - self.u_q) * self.s1_w1[0][w])
            self.eps_c_s1[0] += ((self.vlc_obj.eps_c[0][0] * self.r1_w1[0][w] - self.u_q) * self.s1_w1[0][w])
            self.eps_d_s1[0] += ((self.vlc_obj.eps_d[0][0] * self.r1_w1[0][w] - self.u_q) * self.s1_w1[0][w])
            self.eps_a_s1[1] += ((self.vlc_obj.eps_a[0][1] * self.r2_w1[0][w] - self.u_q) * self.s1_w1[0][w])
            self.eps_b_s1[1] += ((self.vlc_obj.eps_b[0][1] * self.r2_w1[0][w] - self.u_q) * self.s1_w1[0][w])
            self.eps_c_s1[1] += ((self.vlc_obj.eps_c[0][1] * self.r2_w1[0][w] - self.u_q) * self.s1_w1[0][w])
            self.eps_d_s1[1] += ((self.vlc_obj.eps_d[0][1] * self.r2_w1[0][w] - self.u_q) * self.s1_w1[0][w])

            self.eps_a_s2[0] += ((self.vlc_obj.eps_a[1][0] * self.r1_w2[0][w] - self.u_q) * self.s2_w2[0][w])
            self.eps_b_s2[0] += ((self.vlc_obj.eps_b[1][0] * self.r1_w2[0][w] - self.u_q) * self.s2_w2[0][w])
            self.eps_c_s2[0] += ((self.vlc_obj.eps_c[1][0] * self.r1_w2[0][w] - self.u_q) * self.s2_w2[0][w])
            self.eps_d_s2[0] += ((self.vlc_obj.eps_d[1][0] * self.r1_w2[0][w] - self.u_q) * self.s2_w2[0][w])
            self.eps_a_s2[1] += ((self.vlc_obj.eps_a[1][1] * self.r2_w2[0][w] - self.u_q) * self.s2_w2[0][w])
            self.eps_b_s2[1] += ((self.vlc_obj.eps_b[1][1] * self.r2_w2[0][w] - self.u_q) * self.s2_w2[0][w])
            self.eps_c_s2[1] += ((self.vlc_obj.eps_c[1][1] * self.r2_w2[0][w] - self.u_q) * self.s2_w2[0][w])
            self.eps_d_s2[1] += ((self.vlc_obj.eps_d[1][1] * self.r2_w2[0][w] - self.u_q) * self.s2_w2[0][w])

        self.eps_a_s1[0] /= self.hbuf
        self.eps_b_s1[0] /= self.hbuf
        self.eps_c_s1[0] /= self.hbuf
        self.eps_d_s1[0] /= self.hbuf
        self.eps_a_s1[1] /= self.hbuf
        self.eps_b_s1[1] /= self.hbuf
        self.eps_c_s1[1] /= self.hbuf
        self.eps_d_s1[1] /= self.hbuf

        self.eps_a_s2[0] /= self.hbuf
        self.eps_b_s2[0] /= self.hbuf
        self.eps_c_s2[0] /= self.hbuf
        self.eps_d_s2[0] /= self.hbuf
        self.eps_a_s2[1] /= self.hbuf
        self.eps_b_s2[1] /= self.hbuf
        self.eps_c_s2[1] /= self.hbuf
        self.eps_d_s2[1] /= self.hbuf

        self.phi_h_s1[0] = ((self.eps_b_s1[0] + self.eps_d_s1[0]) - (self.eps_a_s1[0] + self.eps_c_s1[0])) / (
                    self.eps_a_s1[0] + self.eps_b_s1[0] + self.eps_c_s1[0] + self.eps_d_s1[0])
        self.phi_h_s1[1] = ((self.eps_b_s1[1] + self.eps_d_s1[1]) - (self.eps_a_s1[1] + self.eps_c_s1[1])) / (
                    self.eps_a_s1[1] + self.eps_b_s1[1] + self.eps_c_s1[1] + self.eps_d_s1[1])
        self.phi_h_s2[0] = ((self.eps_b_s2[0] + self.eps_d_s2[0]) - (self.eps_a_s2[0] + self.eps_c_s2[0])) / (
                    self.eps_a_s2[0] + self.eps_b_s2[0] + self.eps_c_s2[0] + self.eps_d_s2[0])
        self.phi_h_s2[1] = ((self.eps_b_s2[1] + self.eps_d_s2[1]) - (self.eps_a_s2[1] + self.eps_c_s2[1])) / (
                    self.eps_a_s2[1] + self.eps_b_s2[1] + self.eps_c_s2[1] + self.eps_d_s2[1])

        self.theta_R_L[0][0] = self.vlc_obj.translate(self.phi_h_s1[0], -1, 1, (-1 * self.vlc_obj.e_angle),
                                                      self.vlc_obj.e_angle)
        self.theta_R_L[0][1] = self.vlc_obj.translate(self.phi_h_s1[1], -1, 1, (-1 * self.vlc_obj.e_angle),
                                                      self.vlc_obj.e_angle)
        self.theta_R_L[1][0] = self.vlc_obj.translate(self.phi_h_s2[0], -1, 1, (-1 * self.vlc_obj.e_angle),
                                                      self.vlc_obj.e_angle)
        self.theta_R_L[1][1] = self.vlc_obj.translate(self.phi_h_s2[1], -1, 1, (-1 * self.vlc_obj.e_angle),
                                                      self.vlc_obj.e_angle)

        # %%

        self.diff_1 = self.theta_R_L[0][1] - self.theta_R_L[0][0]
        self.t_x_1 = self.vlc_obj.distancecar * (
                    0.5 + (math.sin(self.theta_R_L[0][0]) * math.cos(self.theta_R_L[0][1])) / (math.sin(self.diff_1)))
        self.t_y_1 = self.vlc_obj.distancecar * (
                    (math.cos(self.theta_R_L[0][0]) * math.cos(self.theta_R_L[0][1])) / (math.sin(self.diff_1)))
        # print("Transmitter-1 x pos is : ", t_x_1, ", y pos is : ", t_y_1)

        self.diff_2 = self.theta_R_L[1][1] - self.theta_R_L[1][0]
        self.t_x_2 = self.vlc_obj.distancecar * (
                    0.5 + (math.sin(self.theta_R_L[1][0]) * math.cos(self.theta_R_L[1][1])) / (math.sin(self.diff_2)))
        self.t_y_2 = self.vlc_obj.distancecar * (
                    (math.cos(self.theta_R_L[1][0]) * math.cos(self.theta_R_L[1][1])) / (math.sin(self.diff_2)))
        # print("Transmitter-2 x pos is : ", t_x_2, ", y pos is : ", t_y_2)
        # %%
        # Error calc
        # print("Error of Transmitter-1 position in x:", abs(t_x_1_act-t_x_1),", y:", abs(t_y_1_act-t_y_1))
        # print("Error of Transmitter-2 position in x, y:", abs(t_x_2_act-t_x_2),", y:", abs(t_y_2_act-t_y_2))
        tx_pos = np.array([[self.t_x_1, self.t_y_1], [self.t_x_2, self.t_y_2]])
        return tx_pos

# %%


# %%


