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


    #TODO: terminar funcion de graficar errores
    # def plot_error(self, x, y, title=None, grid=False):
    #     """graficar el error entre dos variables"""

    #     x = np.array(y)

    #     error = np.subtract(y, output2)


    #     error = list(error)

    #     plt.plot(time_list, error)


    #TODO:crear funcion de almacenar data de la simulación

    def show(self):
        """mostrar graficos"""
        plt.show()

# agrizon edicion

