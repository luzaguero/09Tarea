from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

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
    limbajo1 = pendiente[int(N_mc * 0.025)]
    limalto1 = pendiente[int(N_mc * 0.975)]
    limbajo2 = posicion[int(N_mc * 0.025)]
    limalto2 = posicion[int(N_mc * 0.975)]
    print "El intervalo de confianza al 95% para la pendiente es desde {} hasta {}".format(limbajo1, limalto1)
    print "El intervalo de confianza al 95% para el coef de posicion es desde {} hasta {}".format(limbajo2, limalto2)


# Main
c = np.polyfit(datos()[0], datos()[2], 1)
pendiente = montecarlo(datos()[0], datos()[1], datos()[2], datos()[3])[0]
posicion = montecarlo(datos()[0], datos()[1], datos()[2], datos()[3])[1]
bootstrap(pendiente, posicion)

# Grafiquito
x = np.linspace(-100, 500, 600)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.errorbar(datos()[0], datos()[2], xerr=datos()[1], yerr=datos()[3], fmt="go", label="Datos experimentales")
ax1.plot(x, c[1] + x*c[0], color="m", label="Ajuste lineal")

ax1.set_xlabel("Flujo banda i $[10^{-6}Jy]$")
ax1.set_ylabel("Flujo banda z $[10^{-6}Jy]$")

plt.legend(loc='lower right')
plt.show()