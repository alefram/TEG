"""
Funciones para graficar cosas
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

class Logger(object):
    """docstring for Logger"""
    def __init__(self):
        pass

    def plot_trajectory(self, x, y, title=None, grid=False, x_limits=None, y_limits=None):
        """grafica la trajectoria del efector final en el tiempo"""

        if title is not None:
            plt.title(title)

        if grid:
            plt.grid(True)

        if x_limits is not None:
            plt.xlim(x_limits)

        if y_limits is not None:
            plt.ylim(y_limits)

        plt.plot(x, y)


    def plot_error(self, time, error_a, error_b, title=None, grid=False):
        """graficar el error entre dos variables en el tiempo"""
        
        if title is not None:
            plt.title(title)
        
        if grid:
            plt.grid(True)
        

        assert len(error_a) == len(error_b)
        error_a = np.array(error_a)
        error_b = np.array(error_b)

        delta_error = np.subtract(error_a, error_b)
        delta_error = list(delta_error)

        plt.plot(time, delta_error)

    #TODO:crear funcion de almacenar data de la simulación

    def show(self):
        """mostrar graficos"""
        plt.show()

