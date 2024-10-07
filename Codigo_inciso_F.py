import numpy as np
import matplotlib.pyplot as plt

# Parámetros iniciales
J = 1.0  # Constante de acoplamiento
k_B = 1.0  # Constante de Boltzmann
T = 1.0  # Temperatura fija
num_steps = 10000  # Número de pasos de Monte Carlo
N_values = [8, 32, 128]  # Tamaños del sistema
B = 0

# Función de energía total con campo magnético (B = 0 en este caso)
def energia_total(espines, J, B):
    E = -J * np.sum(espines[:-1] * espines[1:]) - B * np.sum(espines)
    return E

# Algoritmo de Metropolis
def metropolis(espines, T, J, B):
    i = np.random.randint(len(espines))
    delta_E = 2 * J * espines[i] * (espines[i-1] + espines[(i+1) % len(espines)]) + 2 * B * espines[i]
    if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
        espines[i] *= -1
    return espines

# Simulación para observar cambios en la magnetización
def simulacion(N, T, num_steps):
    espines = np.full(N, -1) #Configuración inicial (todos los espínes en -1)
    magnetizacion_inst = []
    
    for step in range(num_steps):
        espines = metropolis(espines, T, J, B)
        magnetizacion_inst.append(np.sum(espines))
    
    return magnetizacion_inst

# Realizar simulaciones para diferentes tamaños de sistema
plt.figure(figsize=(12, 6))
for N in N_values:
    magnetizacion_inst = simulacion(N, T, num_steps)
    
    plt.plot(magnetizacion_inst, label=f'N = {N}')
    plt.xlabel("Pasos de Monte Carlo")
    plt.ylabel("Magnetización total")
    plt.title(f"Evolución de la magnetización para T = {T}")
    plt.legend()

# Agregar línea horizontal en y = 0
plt.axhline(y=0, color='k', linestyle='--')

plt.tight_layout()
plt.show()