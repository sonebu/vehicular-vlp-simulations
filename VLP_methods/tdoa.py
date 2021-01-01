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

    def __init__(self, a_m=2, f_m1=40000000, f_m2=25000000, measure_dt=1e-8, vehicle_dt=1e-3, car_dist=1.6):
        self.a_m = a_m
        self.dt = measure_dt
        self.measure_period = vehicle_dt
        self.w1 = 2 * math.pi * f_m1
        self.w2 = 2 * math.pi * f_m2
        self.car_dist = car_dist
        self.t = np.arange(0, vehicle_dt - self.dt, self.dt)

    def estimate(self, delays, H, noise_variance):

        delay1_measured, delay2_measured = self.measure_delay(delays, H, noise_variance)

        # calculate distance differences using d(dist) = delay * c
        v = 3 * 1e8
        ddist1 = np.mean(delay1_measured) * v
        ddist2 = np.mean(delay2_measured) * v

        Y_A = self.car_dist
        D = self.car_dist

        #calculate x,y position of the leading vehicle using eqs. in Robert's method
        if abs(ddist1) > 1e-4 and abs(ddist2) > 1e-4:
            #print("entered to if")
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
            X_A = - math.sqrt(((Y_A ** 2 - 2 * Y_A * Y_B - ddist2 ** 2) / (2 * ddist2)) ** 2 - (Y_B ** 2))
        elif abs(ddist1) <= 1e-4:
            #print("entered to elif")
            Y_B = Y_A / 2 - D
            X_A = - math.sqrt(((2 * D * Y_A - ddist2 ** 2) / (2 - ddist2)) ** 2 - (D - Y_A / 2) ** 2)
        else:
            #print("entered to else")
            Y_B = Y_A / 2
            X_A = - math.sqrt(((2 * Y_A * D + ddist1 ** 2) / (2 * ddist1)) ** 2 - (D + Y_A / 2) ** 2)

#         print("x: ", X_A)
#         print("y: ", (0-Y_B))
        return np.array([[X_A, X_A], [(0-Y_B), (0-Y_B) + self.car_dist]])

    def measure_delay(self, delays, H, noise_variance):
        # after going through ADC at receiver
        delta_delay1 = delays[0][0] - delays[0][1]
        delta_delay2 = delays[1][0] - delays[1][1]

        s1_w1 = H[0][0] * self.a_m * np.cos(self.w1 * (self.t - delta_delay1)) + np.random.normal(0, math.sqrt(noise_variance[0][0]), len(self.t))
        s2_w1 = H[0][1] * self.a_m * np.cos(self.w1 * (self.t)) + np.random.normal(0, math.sqrt(noise_variance[0][1]), len(self.t))

        s1_w2 = H[1][0] * self.a_m * np.cos(self.w2 * (self.t - delta_delay2)) + np.random.normal(0, math.sqrt(noise_variance[1][0]), len(self.t))
        s2_w2 = H[1][1] * self.a_m * np.cos(self.w2 * (self.t)) + np.random.normal(0, math.sqrt(noise_variance[1][1]), len(self.t))

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

        #plt.figure()
        ## plt.xlim(20, 90000)
        ## plt.ylim(0.95e-9, 0.955e-9)
        #plt.plot(delay1_measured[0])
        #plt.plot([0, len(delay1_measured[0]) - 1], [self.delta_delay1, self.delta_delay1], color="red")
        #plt.show()

        #plt.figure()
        #plt.plot(delay2_measured[0])
        #plt.plot([0, len(delay2_measured[0]) - 1], [self.delta_delay2, self.delta_delay2], color="red")
        #plt.show()

        #print(self.delta_delay1)
        #print(np.mean(delay1_measured))

        #print(self.delta_delay2)
        #print(np.mean(delay2_measured))

        return delay1_measured, delay2_measured

