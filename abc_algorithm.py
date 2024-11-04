import numpy as np
import tensorflow as tf
import game
import neural_network as nn

QTD_PARTIDAS_TESTE = 20
PORCENTAGEM_MUTACAO = 0.3
NUMERO_ABELHAS = 20
ITERACOES = 100

def create_random_weights(model):
    # Generate random weights for the network
    return [tf.random.normal(w.shape) for w in model.trainable_weights]

def apply_weights(model, weights):
    # Apply a set of weights to the neural network
    for i, w in enumerate(model.trainable_weights):
        w.assign(weights[i])

def fitness(model, weights):
    apply_weights(model, weights)
    score = 0
    for _ in range(QTD_PARTIDAS_TESTE):  
        score += game.play_game(model)
    return score

def mutate_solution(weights):
    # Create a copy of the weights to mutate
    mutated_weights = [w.numpy() for w in weights]  # Convert tensors to numpy arrays

    # Randomly select indices to mutate
    num_weights = len(mutated_weights)  
    num_mutations = int(num_weights * PORCENTAGEM_MUTACAO)  # Determine how many weights to mutate
    
    # Randomly select indices to mutate
    indices_to_mutate = np.random.choice(num_weights, num_mutations, replace=False)

    # Apply mutation: add a small random value to the selected weights
    for index in indices_to_mutate:
        # Mutate with a small random value, e.g., uniform distribution within [-0.1, 0.1]
        mutation_value = np.random.uniform(-0.1, 0.1, size=mutated_weights[index].shape)
        mutated_weights[index] += mutation_value

    # Convert mutated weights back to tensors
    mutated_weights = [tf.convert_to_tensor(w) for w in mutated_weights]

    return mutated_weights

def optimize_nn(model):
    # # RANDOM WEIGHTS
    # population = [create_random_weights(model) for _ in range(NUMERO_ABELHAS)]
    # best_solution = None
    # best_fitness = -np.inf  # Initialize to a very low value

    # #WEIGHTS FROM SAVED MODEL
    # Load the saved model
    loaded_model = tf.keras.models.load_model("modelv1.h5")
    saved_weights = [tf.convert_to_tensor(w) for w in loaded_model.get_weights()]  # Convert each weight to a tensor
    # #Initialize the population
    population = []
    for i in range(NUMERO_ABELHAS):
        if i == 0:
            # First bee uses the exact saved weights
            population.append(saved_weights)
        else:
            # Other bees use slight mutations of the saved weights
            varied_weights = mutate_solution(saved_weights)
            population.append(varied_weights)
    best_solution = None
    best_fitness = -np.inf  # Initialize to a very low value

    for _ in range(ITERACOES):
        for i, weights in enumerate(population):
            new_weights = mutate_solution(weights)  
            if fitness(model, new_weights) > fitness(model, weights):
                population[i] = new_weights

            # Keep track of the best solution
            current_fitness = fitness(model, population[i])
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = population[i]
            print("-")
            print(f"best_fitness: {best_fitness}")
            print("-")

    # Apply the best solution's weights to the model
    if best_solution is not None:
        apply_weights(model, best_solution)

    return model  # Return the optimized model

# Initialize the neural network
nn_model = nn.create_nn()

# Optimize the neural network using the ABC algorithm
optimized_model = optimize_nn(nn_model)

# Save model
optimized_model.save("modelv2.h5")