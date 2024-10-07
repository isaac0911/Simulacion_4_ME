import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
N = 50         # Número de espines
J = 1.0        # Constante de acoplamiento
k_B = 1.0      # Constante de Boltzmann
T = 0.1        # Temperatura (puedes cambiarla para observar otros comportamientos)
num_steps = 100000  # Número de pasos de Monte Carlo

# Función de energía total
def energia_total(espines, J):
    E = -J * np.sum(espines[:-1] * espines[1:])  # Energía de interacción entre espines vecinos
    return E

# Función para el algoritmo de Metropolis
def metropolis(espines, T, J):
    i = np.random.randint(N)
    delta_E = 2 * J * espines[i] * (espines[i-1] + espines[(i+1) % N])  # Cambio en energía si se invierte el espín
    if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
        espines[i] *= -1  # Invertir espín si se acepta el cambio
    return espines

# Inicializar arrays para energía y magnetización
energia_instantanea = []
magnetizacion_instantanea = []

# Configuración inicial de espines aleatoria
espines = np.random.choice([-1, 1], N)

# Simulaciones de Monte Carlo para la energía y magnetización instantáneas
for step in range(num_steps):
    espines = metropolis(espines, T, J)
    E_instantanea = energia_total(espines, J)
    M_instantanea = np.sum(espines)
    
    # Guardar los resultados en cada paso
    energia_instantanea.append(E_instantanea)
    magnetizacion_instantanea.append(M_instantanea)

# Graficar la energía y magnetización en función del tiempo (número de pasos)
plt.figure(figsize=(12, 5))

# Energía instantánea
plt.subplot(1, 2, 1)
plt.plot(range(num_steps), energia_instantanea, label="Energía instantánea", color="blue")
plt.xlabel("Pasos de Monte Carlo")
plt.ylabel("Energía")
plt.title("Energía instantánea en función del tiempo")
plt.grid(True)
plt.legend()

# Magnetización instantánea
plt.subplot(1, 2, 2)
plt.plot(range(num_steps), magnetizacion_instantanea, label="Magnetización instantánea", color="red")
plt.xlabel("Pasos de Monte Carlo")
plt.ylabel("Magnetización")
plt.title("Magnetización instantánea en función del tiempo")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()