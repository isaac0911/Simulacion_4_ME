import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
N = 50          # Número de espines
J = 1.0         # Constante de acoplamiento
k_B = 1.0       # Constante de Boltzmann
num_steps = 10000  # Número de pasos de Monte Carlo por temperatura
temperaturas = np.linspace(0.5, 5.0, 20)  # Temperaturas a explorar

# Inicializar arrays para guardar resultados
energia_media = []
magnetizacion_media = []
susceptibilidad = []
capacidad_calorifica = []
energia_exacta = []

# Función de energía total
def energia_total(espines, J):
    E = -J * np.sum(espines[:-1] * espines[1:])  # Energía de interacción entre espines vecinos
    return E

# Función para el algoritmo de Metropolis
def metropolis(espines, T, J):
    for _ in range(N):  # Un paso de Monte Carlo por spin
        i = np.random.randint(N)
        delta_E = 2 * J * espines[i] * (espines[i-1] + espines[(i+1) % N])  # Cambio en energía si se invierte el espín
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
            espines[i] *= -1  # Invertir espín si se acepta el cambio
    return espines

# Simulaciones a diferentes temperaturas
for T in temperaturas:
    espines = np.random.choice([-1, 1], N)  # Configuración inicial aleatoria
    E_total = []
    M_total = []

    # Simulaciones de Monte Carlo
    for _ in range(num_steps):
        espines = metropolis(espines, T, J)
        E_total.append(energia_total(espines, J))
        M_total.append(np.sum(espines))
    
    # Calcular los promedios y fluctuaciones
    energia_promedio = np.mean(E_total)
    magnetizacion_promedio = np.mean(np.abs(M_total))
    chi = (np.var(M_total)) / (k_B * T)  # Susceptibilidad magnética
    C = (np.var(E_total)) / (k_B * T**2)  # Capacidad calorífica

    # Guardar los resultados
    energia_media.append(energia_promedio)
    magnetizacion_media.append(magnetizacion_promedio)
    susceptibilidad.append(chi)
    capacidad_calorifica.append(C)

    # Comparación con la energía exacta
    energia_exacta.append(-N * np.tanh(1 / T))

# Graficar los resultados
plt.figure(figsize=(12, 8))

# Energía media y comparación con la solución exacta
plt.subplot(2, 2, 1)
plt.plot(temperaturas, energia_media, label="Energía (Metropolis)")
plt.plot(temperaturas, energia_exacta, label="Energía Exacta", linestyle='dashed')
plt.xlabel("Temperatura (T)")
plt.ylabel("Energía media")
plt.legend()

# Magnetización media
plt.subplot(2, 2, 2)
plt.plot(temperaturas, magnetizacion_media, label="Magnetización")
plt.xlabel("Temperatura (T)")
plt.ylabel("Magnetización media")
plt.legend()

# Susceptibilidad
plt.subplot(2, 2, 3)
plt.plot(temperaturas, susceptibilidad, label="Susceptibilidad")
plt.xlabel("Temperatura (T)")
plt.ylabel("Susceptibilidad")
plt.legend()

# Capacidad calorífica
plt.subplot(2, 2, 4)
plt.plot(temperaturas, capacidad_calorifica, label="Capacidad Calorífica")
plt.xlabel("Temperatura (T)")
plt.ylabel("Capacidad Calorífica")
plt.legend()

plt.tight_layout()
plt.show()