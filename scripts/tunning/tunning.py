import numpy as np
from sklearn.model_selection import StratifiedKFold
from random import uniform
from ..sets_builder import set_picker as sp
from ..data_processing import analyzer


# Random seed for reproducibility
np.random.seed(42)

# Create a SetPicker object
set_picker = sp.SetPicker('demo_data/data_a', 'demo_data/data_b')

TA = set_picker.get_a_set() # Total A set

z_percentage = 0.7
z_index = int(len(TA) * z_percentage)

Z = TA[:z_index]
z_matrices_data = set_picker.get_z_matrices(Z)
z_matrices = [z_matrices_data[0]['distance_matrix'], z_matrices_data[1]['distance_matrix'], z_matrices_data[2]['distance_matrix']]

names_dict = {}
A_SET = TA[z_index:]
B_SET = set_picker.get_b_set()

count = 0
for name in A_SET:
    names_dict[count] = name
    count += 1
for name in B_SET:
    names_dict[count] = name
    count += 1

A = []
for name in A_SET:
    A.append([key for key, value in names_dict.items() if value == name][0])

B = []
for name in B_SET:
    B.append([key for key, value in names_dict.items() if value == name][0])

C = A + B

A = np.array(A)
B = np.array(B)
C = np.array(C)

labels_dic = set_picker.get_labels()

def test(z_data, alpha, beta, gamma, threshold, element, names_dict):
    global set_picker
    global Z
    xi_name = names_dict[element]

    xi_matrix = set_picker.get_xi_matrix(Z, xi_name)
    filenames = xi_matrix[0]['filenames']
    xi_matrix = [xi_matrix[0]['distance_matrix'], xi_matrix[1]['distance_matrix'], xi_matrix[2]['distance_matrix']]

    return analyzer.get_prediction_values(alpha, beta, gamma, threshold, z_data["min_value"], xi_matrix, filenames, xi_name)

def fitness(individual, x_train, y_train, z_matrices, names_dict, labels_dic):
    alpha, beta, gamma, threshold = individual

    # Convert threshold from [0, 1] to [0, 100]
    threshold = int(threshold * 100)

    z_data = analyzer.get_distance_data(alpha, beta, gamma, threshold, z_matrices)

    correct = 0
    for i, element in enumerate(x_train):
        prediction = test(z_data, alpha, beta, gamma, threshold, element, names_dict)
        val = prediction['pass']
        # If val is True, the prediction is A, otherwise it is B
        val = 'a' if val else 'b'
        if val == labels_dic[names_dict[element]]:
            correct += 1

    print(f"Correct: {correct} / {len(x_train)}")
    # Fitness as accuracy
    return correct / len(x_train) 

# Función del Micro-Genetic Algorithm
def micro_ga(X_train, y_train, z_matrices, names_dict, labels_dic, generations=10, population_size=2):
    # Inicialización de población
    population = [np.array([uniform(0, 1), uniform(0, 1), uniform(0, 1), uniform(0, 1)]) for _ in range(population_size)]
    
    for generation in range(generations):
        # Evaluar aptitud de cada individuo
        fitness_scores = [fitness(ind, X_train, y_train, z_matrices, names_dict, labels_dic) for ind in population]
        
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

# Cross-Validation Estratificada
skf = StratifiedKFold(n_splits=10)

# Convertir diccionario de etiquetas a un arreglo
labels = np.array([0]*len(A) + [1]*len(B))  # Etiquetas: 0 para A, 1 para B

for fold, (train_idx, test_idx) in enumerate(skf.split(C, labels)):
    print(f"Fold {fold + 1}")
    X_train, X_test = C[train_idx], C[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    # Entrenar Micro-GA en los datos de entrenamiento
    best_parameters = micro_ga(X_train, y_train, z_matrices, names_dict, labels_dic)
    print(f"Mejores parámetros (alpha, beta, gamma, threshold): {best_parameters}")
    
    # Evaluar en los datos de prueba
    accuracy = fitness(best_parameters, X_test, y_test, z_matrices, names_dict, labels_dic)
    print(f"Precisión en el conjunto de prueba: {accuracy:.2f}\n")