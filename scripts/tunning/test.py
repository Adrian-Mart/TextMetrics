import numpy as np
from sklearn.model_selection import StratifiedKFold
from random import uniform

# Datos simulados (50 elementos para A y 50 para B)
# Cada elemento tiene tres valores: W_i, W_l y W_t
np.random.seed(42)  # Para reproducibilidad
A = np.random.rand(50, 3)  # Clase A
B = np.random.rand(50, 3)  # Clase B
C = np.vstack((A, B))  # Conjunto completo
print(C.shape)  # (100, 3)
labels = np.array([0]*50 + [1]*50)  # Etiquetas: 0 para A, 1 para B

# Función para evaluar la aptitud de un individuo
def fitness(individual, X_train, y_train):
    alpha, beta, gamma, threshold = individual
    correct = 0
    for i, element in enumerate(X_train):
        W_i, W_l, W_t = element
        result = alpha * W_i + beta * W_l + gamma * W_t
        prediction = 0 if result >= threshold else 1  # 0 -> pertenece a A, 1 -> no pertenece
        if prediction == y_train[i]:
            correct += 1
    return correct / len(X_train)  # Precisión como métrica de aptitud

# Función del Micro-Genetic Algorithm
def micro_ga(X_train, y_train, generations=50, population_size=5):
    # Inicialización de población
    population = [np.array([uniform(0, 1), uniform(0, 1), uniform(0, 1), uniform(0, 1)]) for _ in range(population_size)]
    
    for generation in range(generations):
        # Evaluar aptitud de cada individuo
        fitness_scores = [fitness(ind, X_train, y_train) for ind in population]
        
        # Seleccionar al mejor individuo (elitismo)
        best_individual = population[np.argmax(fitness_scores)]
        
        # Generar nueva población mediante cruza y mutación
        new_population = [best_individual]  # Mantener al mejor individuo
        while len(new_population) < population_size:
            # Cruza de dos padres
            # # Selección de individuos usando índices
            indices = np.random.choice(len(population), 2, replace=False)
            parent1, parent2 = population[indices[0]], population[indices[1]]
            crossover_point = np.random.randint(1, 4)  # Punto de cruza
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            
            # Mutación
            mutation_chance = 0.2
            if uniform(0, 1) < mutation_chance:
                mutation_idx = np.random.randint(0, 4)
                child[mutation_idx] = uniform(0, 1)  # Nueva mutación en el rango [0, 1]
            
            new_population.append(child)
        
        population = new_population  # Actualizar población
    
    return best_individual  # Devolver el mejor conjunto de parámetros

# Cross-Validation
skf = StratifiedKFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(skf.split(C, labels)):
    print(f"Fold {fold + 1}")
    X_train, X_test = C[train_idx], C[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    # Entrenar Micro-GA en los datos de entrenamiento
    best_parameters = micro_ga(X_train, y_train)
    print(f"Mejores parámetros (alpha, beta, gamma, threshold): {best_parameters}")
    
    # Evaluar en los datos de prueba
    accuracy = fitness(best_parameters, X_test, y_test)
    print(f"Precisión en el conjunto de prueba: {accuracy:.2f}\n")