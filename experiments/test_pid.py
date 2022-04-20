import time
from scipy.interpolate import BSpline, make_interp_spline #  Switched to BSpline
import numpy as np
import matplotlib.pyplot as plt

class PID:

    def __init__(self, P=0.0, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.0
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.reset()


    def reset(self):
        """Resetear coeficientes y calculos"""

        self.r_t = 0.0
        self.P = 0.0
        self.I = 0.0
        self.D = 0.0
        self.last_e_t = 0.0
        self.int_error = 0.0
        self.overshoot_guard = 20.0
        self.u_t = 0.0


    def update(self, y_t, current_time=None):
        """
            actualizar valores del PID dado la señal de salida

            u(t) = Kp e(t) + Ki integral(e(t)) + Kd d/dt (e(t)) = P + I + D
        """

        e_t = self.r_t - y_t

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_e_t = e_t - self.last_e_t

        if (delta_time >= self.sample_time):
            self.P = e_t
            self.I += e_t * delta_time

            if (self.I < -self.overshoot_guard):
                self.I = -self.overshoot_guard
            elif (self.I > self.overshoot_guard):
                self.I = self.overshoot_guard

            self.D = 0.0

            if delta_time > 0:
                self.D = delta_e_t / delta_time

            #recordar ultimo valores para calculo final
            self.last_time = self.current_time
            self.last_e_t = e_t

            self.u_t = (self.Kp * self.P) + (self.Ki * self.I) + (self.Kd * self.D)


    def set_Kp(self, proportional_gain):
        """Ganancia proporcional"""
        self.Kp = proportional_gain

    def set_Ki(self, integral_gain):
        """Ganancia Integral"""
        self.Ki = integral_gain

    def set_Kd(self, derivative_gain):
        """Ganancia derivativa"""
        self.Kd = derivative_gain

    def setWindup(self, overshoot):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """

        self.overshoot_guard = overshoot

    def set_sample_time(self, sample_time):
        """tiempo de muestreo"""

        self.sample_time = sample_time


def main():

    L = 50
    pid = PID(P=0.7,I=1.5, D=0.001)

    pid.r_t = 0.0
    pid.set_sample_time(0.01)

    END = L
    y_t = 0

    y_t_list  = []
    time_list = []
    r_t_list = []

    for i in range(1, END):
        pid.update(y_t)
        u_t = pid.u_t
        if pid.r_t > 0:
            y_t += (u_t - (1/i))
        if  i > 9:
            pid.r_t = 1

        time.sleep(0.02)

        y_t_list.append(y_t)
        r_t_list.append(pid.r_t)
        time_list.append(i)

    time_sm = np.array(time_list)
    time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)

    helper_x3 = make_interp_spline(time_list, y_t_list)
    feedback_smooth = helper_x3(time_smooth)

    plt.plot(time_smooth, feedback_smooth)
    plt.plot(time_list, r_t_list)
    plt.xlim((0, L))
    plt.ylim((min(y_t_list)-0.5, max(y_t_list)+0.5))
    plt.xlabel('time (s)')
    plt.ylabel('PID (PV)')
    plt.title('TEST PID')

    plt.ylim((1-0.5, 1+0.5))

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
