import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
N = 50  # Número de espines
J = 1.0  # Constante de acoplamiento
k_B = 1.0  # Constante de Boltzmann
num_steps = 10000  # Número de pasos de Monte Carlo
temperaturas = np.linspace(0.5, 5, 20)  # Temperaturas para barrer
campos_magneticos = [0.1, 0.5, 1.0]  # Valores de campo magnético

# Función de energía total con campo magnético
def energia_total(espines, J, B):
    E = -J * np.sum(espines[:-1] * espines[1:]) - B * np.sum(espines)
    return E

# Algoritmo de Metropolis
def metropolis(espines, T, J, B):
    i = np.random.randint(N)
    delta_E = 2 * J * espines[i] * (espines[i-1] + espines[(i+1) % N]) + 2 * B * espines[i]
    if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
        espines[i] *= -1
    return espines

# Inicializar arrays para energía y magnetización
def simulacion(B):
    energia_media = []
    magnetizacion_media = []
    
    for T in temperaturas:
        energia_inst = []
        magnetizacion_inst = []
        
        espines = np.random.choice([-1, 1], N)  # Configuración inicial aleatoria
        
        for step in range(num_steps):
            espines = metropolis(espines, T, J, B)
            E_instantanea = energia_total(espines, J, B)
            M_instantanea = np.sum(espines)
            energia_inst.append(E_instantanea)
            magnetizacion_inst.append(M_instantanea)
        
        # Calcular los promedios de energía y magnetización
        energia_media.append(np.mean(energia_inst) / N)
        magnetizacion_media.append(np.mean(magnetizacion_inst) / N)
    
    return energia_media, magnetizacion_media

# Realizar simulaciones para los diferentes campos magnéticos
for B in campos_magneticos:
    energia_media, magnetizacion_media = simulacion(B)
    
    # Graficar energía en función de la temperatura
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(temperaturas, energia_media, label=f'B = {B}')
    plt.xlabel("Temperatura")
    plt.ylabel("Energía media por espín")
    plt.title("Energía en función de la temperatura")
    plt.legend()

    # Graficar magnetización en función de la temperatura
    plt.subplot(1, 2, 2)
    plt.plot(temperaturas, magnetizacion_media, label=f'B = {B}')
    plt.xlabel("Temperatura")
    plt.ylabel("Magnetización media por espín")
    plt.title("Magnetización en función de la temperatura")
    plt.legend()

plt.tight_layout()
plt.show()