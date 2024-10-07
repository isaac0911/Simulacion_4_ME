import numpy as np

def metropolis_ising_1D(N, J, B, T, num_steps):
    # Inicialización de espines aleatorios
    spins = np.random.choice([-1, 1], size=N)
    
    # Constante de Boltzmann (suponemos kB = 1 para simplificar)
    kB = 1
    
    # Definir función para calcular el cambio de energía
    def energy_change(spins, i):
        # Condiciones de frontera periódicas
        left_neighbor = spins[i-1] if i > 0 else spins[N-1]
        right_neighbor = spins[i+1] if i < N-1 else spins[0]
        return 2 * J * spins[i] * (left_neighbor + right_neighbor)
    
    # Inicializar listas para acumular datos
    energy_list = []
    magnetization_list = []

    for step in range(num_steps):
        for _ in range(N):  # Un paso Monte Carlo por vuelta (N intentos)
            i = np.random.randint(0, N)  # Seleccionar espín aleatoriamente
            dE = energy_change(spins, i) + 2 * B * spins[i]  # Cambio de energía

            # Decidir si aceptar el cambio
            if dE < 0 or np.random.rand() < np.exp(-dE / (kB * T)):
                spins[i] *= -1  # Cambiar el espín
        
        # Calcular energía total y magnetización en cada paso
        energy = -J * sum(spins[i] * spins[(i+1) % N] for i in range(N)) - B * sum(spins)
        magnetization = sum(spins)
        
        # Acumular energía y magnetización
        energy_list.append(energy)
        magnetization_list.append(magnetization)
    
    # Convertir las listas a arrays para cálculos más fáciles
    energy_array = np.array(energy_list)
    magnetization_array = np.array(magnetization_list)
    
    # Promedios y cuadrados de energía y magnetización
    E_mean = np.mean(energy_array)
    E2_mean = np.mean(energy_array**2)
    M_mean = np.mean(magnetization_array)
    M2_mean = np.mean(magnetization_array**2)
    
    # Cálculo de la susceptibilidad magnética y la capacidad calorífica
    susceptibility = (M2_mean - M_mean**2) / (kB * T)
    heat_capacity = (E2_mean - E_mean**2) / (kB * T**2)
    
    return E_mean, M_mean, susceptibility, heat_capacity

# Parámetros del modelo
N = 100  # Número de espines
J = 1.0  # Interacción entre espines
B = 0.0  # Campo externo
T = 1.0  # Temperatura
num_steps = 10000  # Número de pasos de Monte Carlo

# Ejecutar simulación
energia_media, magnetizacion_media, susceptibilidad, capacidad_calorifica = metropolis_ising_1D(N, J, B, T, num_steps)

# Imprimir resultados
print("Energía media:", energia_media)
print("Magnetización media:", magnetizacion_media)
print("Susceptibilidad magnética:", susceptibilidad)
print("Capacidad calorífica:", capacidad_calorifica)