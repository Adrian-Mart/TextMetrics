import numpy as np
from sklearn.model_selection import StratifiedKFold
from random import uniform
from ..sets_builder import set_picker as sp
from ..data_processing import analyzer
from tqdm import tqdm
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Micro-Genetic Algorithm for parameter tuning.')
parser.add_argument('-g', '--generations', type=int, default=10, help='Number of generations')
parser.add_argument('-p', '--population-size', type=int, default=50, help='Population size')
parser.add_argument('-f', '--folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('-zp', '--z-percentage', type=float, default=0.9, help='Percentage of Z set')
parser.add_argument('-m', '--mutations', type=float, default=0.2, help='Mutation rate')

args = parser.parse_args()

# Random seed for reproducibility
np.random.seed(42)

# Create a SetPicker object
set_picker = sp.SetPicker('demo_data', 'demo_data/data_a', 'demo_data/data_b')

TA = set_picker.get_a_set() # Total A set

z_percentage = args.z_percentage
z_index = int(len(TA) * z_percentage)

Z = TA[:z_index]
set_picker.set_z_set(Z)

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

labels_dict = set_picker.get_labels()

def fitness(individual, x_train, names_dict, labels_dict, print_=False):
    alpha, beta, gamma = individual

    predictions = set_picker.get_predictions(alpha, beta, gamma)

    correct = 0
    for i, element in enumerate(x_train):
        name = names_dict[element]
        prediction = predictions[name]
        # If val is True, the prediction is A, otherwise it is B
        prediction = 'a' if prediction else 'b'
        if prediction == labels_dict[name]:
            correct += 1
    # Fitness as accuracy
    return correct / len(x_train) 

def roulette_selection(population, fitness_scores):
    # Convert fitness to probabilities
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]

    # Choose two parents based on probabilities
    indices = np.random.choice(len(population), 2, replace=False, p=probabilities)
    return population[indices[0]], population[indices[1]]

def maintain_diversity(population, fitness_scores, num_new=5):
    # Generate new random individuals
    new_individuals = [np.array([uniform(0, 1), uniform(0, 1), uniform(0, 1)]) for _ in range(num_new)]

    # Replace the worst individuals with the new ones
    sorted_indices = np.argsort(fitness_scores)  # Sort by fitness
    for idx in sorted_indices[:num_new]:
        population[idx] = new_individuals.pop(0)

    return population

def adaptive_mutation_rate(generation, max_generations, initial_rate=0.2, final_rate=0.05):
    # Decrease the mutation rate linearly
    return initial_rate - (generation / max_generations) * (initial_rate - final_rate)

# Micro-Genetic Algorithm function
def micro_ga(X_train, names_dict, labels_dict, generations=10, population_size=50, mutation_rate=0.2):
    # Population initialization
    population = [np.array([uniform(0, 1), uniform(0, 1), uniform(0, 1)]) for _ in range(population_size)]
    
    for generation in tqdm(range(generations), desc="Generations"):
        # Evaluate fitness of each individual
        fitness_scores = [fitness(ind, X_train, names_dict, labels_dict) for ind in population]
        population = maintain_diversity(population, fitness_scores, num_new=5)
        
        # Select the best individual (elitism)
        best_individual = population[np.argmax(fitness_scores)]
        
        # Generate new population through crossover and mutation
        new_population = [best_individual]  # Keep the best individual
        while len(new_population) < population_size:
            # Crossover of two parents
            parent1, parent2 = roulette_selection(population, fitness_scores)
            crossover_point = np.random.randint(1, 4)  # Crossover point
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            
            current_mutation_rate = adaptive_mutation_rate(generation, generations, mutation_rate)
            if uniform(0, 1) < current_mutation_rate:
                mutation_idx = np.random.randint(0, 3)
                child[mutation_idx] = uniform(0, 1)  # Apply mutation
            
            new_population.append(child)
        
        population = new_population  # Update population
    
    return best_individual  # Return the best set of parameters

def weighted_average(parameters):
    accuracies = [accuracy for parameters, accuracy in parameters]
    parameters = np.array([parameters for parameters, accuracy in parameters])
    total_accuracy = np.sum(accuracies)
    weighted_parameters = [0, 0, 0]
    index = 0
    for i in parameters:
        weighted_parameters += i * accuracies[index]
        index += 1
    weighted_parameters /= total_accuracy
    return weighted_parameters


# Stratified Cross-Validation
skf = StratifiedKFold(n_splits=args.folds)

# Convert label dictionary to an array
labels = np.array([0]*len(A) + [1]*len(B))  # Labels: 0 for A, 1 for B

best_parameters_list = []
for fold, (train_idx, test_idx) in enumerate(skf.split(C, labels)):
    print(f"Fold {fold + 1}")
    X_train, X_test = C[train_idx], C[test_idx]
    
    # Train Micro-GA on the training data
    best_parameters = micro_ga(X_train, names_dict, labels_dict, generations=args.generations, population_size=args.population_size, mutation_rate=args.mutations)
    print(f"Best parameters (alpha, beta, gamma): {best_parameters}")
    
    # Evaluate on the test data
    accuracy = fitness(best_parameters, X_test, names_dict, labels_dict, True)
    print(f"Accuracy on the test set: {accuracy:.2f}\n")

    best_parameters_list.append((best_parameters, accuracy))

weighted_parameters = weighted_average(best_parameters_list)
print(f"Weighted best parameters: {weighted_parameters}")

# Make a test using B as the test set
set_picker = sp.SetPicker('demo_data', 'demo_data/data_a', 'demo_data/data_b')
set_picker.set_z_set(TA)

# Test the weighted parameters
accuracy = fitness(weighted_parameters, B, names_dict, labels_dict, True)

print(f"Accuracy on the test set: {accuracy:.2f}\n")
