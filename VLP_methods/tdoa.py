from cache.VLC_init import *

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


class TDoA:

    def __init__(self, a_m=2, f_m1=40000000, f_m2=25000000, measure_dt=1e-8, vehicle_dt=1e-3, car_dist=1.6, c=3e8):
        """

        :param a_m: initial power of the transmitted signal
        :param f_m1: tone/frequency of the signal from trx_1
        :param f_m2: tone/frequency of the signal from trx_2
        :param measure_dt: time increment to measure the received signal
        :param vehicle_dt: time between vehicle position measurements
        :param car_dist: distance between two headlights/taillights of the car
        :param c: speed of light
        """
        self.a_m = a_m
        self.dt = measure_dt
        self.measure_period = vehicle_dt
        self.w1 = 2 * math.pi * f_m1
        self.w2 = 2 * math.pi * f_m2
        self.car_dist = car_dist
        self.t = np.arange(0, vehicle_dt - self.dt, self.dt)
        self.c = c

    def estimate(self, delays, H, noise_variance):
        """
        Implements the method of Roberts et al. using received signal
        :param delays: actual delay values of transmitted signals, 2*2 matrix, first index is for tx second is for rx
        :param H: attenuation on the signal, 2*2 matrix, first index is for tx second is for rx
        :param noise_variance: AWGN variance values for each signal, 2*2 matrix, first index is for tx second is for rx
        :return: estimated positions of the transmitting vehicle
        """
        #calculate measured delay using attenuation and noise on the signal
        delay1_measured, delay2_measured = self.measure_delay(delays, H, noise_variance)

        # calculate distance differences using d(dist) = delay * c
        v = self.c
        ddist1 = np.mean(delay1_measured) * v
        ddist2 = np.mean(delay2_measured) * v

        #following notations such as Y_A, D, A, and B are in line with the paper itself
        Y_A = self.car_dist
        D = self.car_dist

        #calculate x,y position of the leading vehicle using eqs. in Robert's method
        if abs(ddist1) > 1e-4 and abs(ddist2) > 1e-4:
            A = Y_A ** 2 * (1 / (ddist1 ** 2) - 1 / (ddist2 ** 2))

            B1 = (-(Y_A ** 3) + 2 * (Y_A ** 2) * D + Y_A * (ddist1 ** 2)) / (ddist1 ** 2)
            B2 = (-(Y_A ** 3) + Y_A * (ddist2 ** 2)) / (ddist2 ** 2)
            B = B1 - 2 * D - B2

            C1 = ((Y_A ** 4) + 4 * (D ** 2) * (Y_A ** 2) + (ddist1 ** 4) - 4 * D * (Y_A ** 3) - 2 * (Y_A ** 2) * (
                        ddist1 ** 2) + 4 * D * Y_A * (ddist1 ** 2)) / (4 * (ddist1 ** 2))
            C2 = ((Y_A ** 4) + (ddist2 ** 4) - 2 * (Y_A ** 2) * (ddist2 ** 2)) / (4 * (ddist2 ** 2))
            C = C1 - D ** 2 - C2

            if ddist1 * ddist2 > 0:
                Y_B = (- B - math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            else:
                Y_B = (- B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
            if ((Y_A ** 2 - 2 * Y_A * Y_B - ddist2 ** 2) / (2 * ddist2)) ** 2 - (Y_B ** 2) < 0:
                # since assumes parallel, fails to find close delays -> negative in srqt
                return np.array([[float('NaN'), float('NaN')], [float('NaN'), float('NaN')]])
            X_A = - math.sqrt(((Y_A ** 2 - 2 * Y_A * Y_B - ddist2 ** 2) / (2 * ddist2)) ** 2 - (Y_B ** 2))
        elif abs(ddist1) <= 1e-4:
            Y_B = Y_A / 2 - D
            if ((2 * D * Y_A - ddist2 ** 2) / (2 - ddist2)) ** 2 - (D - Y_A / 2) ** 2 < 0:
                # since assumes parallel, fails to find close delays -> negative in srqt
                return np.array([[float('NaN'), float('NaN')], [float('NaN'), float('NaN')]])
            X_A = - math.sqrt(((2 * D * Y_A - ddist2 ** 2) / (2 - ddist2)) ** 2 - (D - Y_A / 2) ** 2)
        else:
            Y_B = Y_A / 2
            if ((2 * Y_A * D + ddist1 ** 2) / (2 * ddist1)) ** 2 - (D + Y_A / 2) ** 2 < 0:
                # since assumes parallel, fails to find close delays -> negative in srqt
                return np.array([[float('NaN'), float('NaN')], [float('NaN'), float('NaN')]])
            X_A = - math.sqrt(((2 * Y_A * D + ddist1 ** 2) / (2 * ddist1)) ** 2 - (D + Y_A / 2) ** 2)
        return np.array([[X_A, X_A], [(0-Y_B), (0-Y_B) + self.car_dist]])

    def measure_delay(self, delays, H, noise_variance):
        """
        creates the received signal using input parameters and calculates delay measured by rx
        :param delays:
        :param H:
        :param noise_variance:
        :return: a tuple where first element is the delay difference between the received signals sent by tx1,
        second element is the delay difference between the received signals sent by tx1
        """
        # after going through ADC at receiver
        delta_delay1 = delays[0][0] - delays[0][1]
        delta_delay2 = delays[1][0] - delays[1][1]

        #create received signals
        s1_w1 = H[0][0] * self.a_m * np.cos(self.w1 * (self.t - delta_delay1)) + np.random.normal(0, math.sqrt(noise_variance[0][0]), len(self.t))
        s2_w1 = H[0][1] * self.a_m * np.cos(self.w1 * (self.t)) + np.random.normal(0, math.sqrt(noise_variance[0][1]), len(self.t))

        s1_w2 = H[1][0] * self.a_m * np.cos(self.w2 * (self.t - delta_delay2)) + np.random.normal(0, math.sqrt(noise_variance[1][0]), len(self.t))
        s2_w2 = H[1][1] * self.a_m * np.cos(self.w2 * (self.t)) + np.random.normal(0, math.sqrt(noise_variance[1][1]), len(self.t))

        # take fourier transform
        s1_w1_fft = np.fft.fft(s1_w1)
        s2_w1_fft = np.fft.fft(s2_w1)

        #remove left half for both singals
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

        #multiply the signals to obtain delay difference
        direct_mix1 = np.multiply(s1_w1_upperSideband, s2_w1_upperSideband.conj())
        delay1_measured = np.angle(direct_mix1) / self.w1

        direct_mix2 = np.multiply(s1_w2_upperSideband, s2_w2_upperSideband.conj())
        delay2_measured = np.angle(direct_mix2) / self.w2

        return delay1_measured, delay2_measured

