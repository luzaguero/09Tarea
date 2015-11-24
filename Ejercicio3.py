from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

np.random.seed(1911)

def datos():
    data = np.loadtxt("data/DR9Q.dat", usecols=(80, 81, 82, 83))
    banda_i = data[:, 0] * 3.631
    error_i = data[:, 1] * 3.631
    banda_z = data[:, 2] * 3.631
    error_z = data[:, 3] * 3.631
    return banda_i, error_i, banda_z, error_z

def montecarlo(banda_i, error_i, banda_z, error_z):
    '''
    Simulacion de montecarlo
    '''
    N_mc = 100
    posicion = np.zeros(N_mc)
    pendiente = np.zeros(N_mc)
    m = len(banda_i)
    for n in range(N_mc):
        M = np.random.normal(0, 1, size=m)
        i_n = banda_i + error_i * M
        z_n = banda_z + error_z * M
        pendiente[n], posicion[n] = np.polyfit(i_n, z_n, 1)

    return pendiente, posicion

def bootstrap(pendiente, posicion):
    '''
    Bootstrap para encontrar el intervalo de confianza al 95%
    '''
    N_mc = 100
    pendiente = np.sort(pendiente)
    posicion = np.sort(posicion)
    limite_bajo_1 = pendiente[int(N_mc * 0.025)]
    limite_alto_1 = pendiente[int(N_mc * 0.975)]
    limite_bajo_2 = posicion[int(N_mc * 0.025)]
    limite_alto_2 = posicion[int(N_mc * 0.975)]
    print "El intervalo de confianza al 95% para la pendiente es desde {} hasta {}".format(limite_bajo_1, limite_alto_1)
    print "El intervalo de confianza al 95% para el coef de posicion es desde {} hasta {}".format(limite_bajo_2, limite_alto_2)

