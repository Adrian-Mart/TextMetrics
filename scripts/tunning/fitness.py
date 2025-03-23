def test(z_train, alpha, beta, gamma, threshold, element):
    pass

def fitness(individual, x_train, z_train, labels):
    alpha, beta, gamma, threshold = individual

    # Convert threshold from [0, 1] to [0, 100]
    threshold = int(threshold * 100)

    correct = 0
    for i, element in enumerate(x_train):
        prediction = test(z_train, alpha, beta, gamma, threshold, element)
        if prediction == labels[i]:
            correct += 1

    # Fitness as accuracy
    return correct / len(x_train)  