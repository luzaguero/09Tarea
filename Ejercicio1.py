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

# Setup
distancia = cargar_datos()[0]
velocidad = cargar_datos()[1]

# Main
H_01 = minimizar(distancia, velocidad, 100.)[0]
H_02 = 1/minimizar(velocidad, distancia, 100.)[0]
H_prom = (H_01 + H_02) / 2

bootstrap()
print "Valor estimado H_0 =", H_prom[0]

# Grafiquito
fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(111)

ax1.plot(distancia, velocidad, 'bo', label="Datos experimentales")
ax1.plot(distancia, H_prom * distancia, 'y-', label="Promedio $H_0$")
ax1.plot(distancia, H_01 * distancia, 'g--', label="$v = H_0*D$")
ax1.plot(distancia, H_02 * distancia, 'r--', label="$D = v/H_0$")

ax1.set_xlim([-0.5, 2.5])
ax1.set_xlabel("Distancia $[Mpc]$")
ax1.set_ylabel("Velocidad $[km/s]$")

plt.legend(loc='lower right')
plt.show()

