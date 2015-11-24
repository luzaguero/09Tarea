from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

np.random.seed(1911)

def cargar_datos():
    datos = np.loadtxt("data/hubble_original.dat")
    d = datos[:, 0]
    v = datos[:, 1]
    return d, v, datos

def funcion_modelo(parametro, variable):
    '''
    2 modelos posibles, con distintos resultados.
    v = Ho * d #1
    d = (1 / Ho) * v (Caso 2) #2
    '''
    return parametro * variable

def minimizar(x, y, N):
    '''
    x,y = distancia, velocidad #1
    x,y = velocidad, distancia #2
    '''
    funcion = lambda i, xdata, ydata: ydata - funcion_modelo(i, xdata)
    i = np.random.randint(N)
    resultado = leastsq(funcion, i, args=(x, y))

    return resultado[:2]

def bootstrap():
    '''
    Bootstrap para encontrar el intervalo de confianza al 95%
    '''
    N_boot = 10000
    H = np.zeros(N_boot)
    for i in range(N_boot):
        H1, e1 = minimizar(distancia, velocidad, 100.)
        h2, e2 = minimizar(velocidad, distancia, 100.)
        H2 = 1 / h2
        Hprom = (H1 + H2) / 2
        H[i] = Hprom
    H = np.sort(H)
    lim_inf = H[int(N_boot * 0.025)]
    lim_sup = H[int(N_boot * 0.975)]
    print "El intervalo de confianza al 95% es desde {} hasta {}".format(lim_inf, lim_sup)



