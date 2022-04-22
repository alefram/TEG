import time
from scipy.interpolate import BSpline, make_interp_spline #  Switched to BSpline
import numpy as np
import matplotlib.pyplot as plt
from RobotEnv.tools.controllers import PID


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
