from popop import POPOP

import numpy as np


def initialize_population(n, pop_size):
    pop = []
    for i in range(pop_size):
        ind = np.arange(1, n+1)
        np.random.shuffle(ind[1:])
        pop.append(ind)

    return np.array(pop)


def ordered_crossover(pop, pool_over_pop_size):
    num_individuals = pop.shape[0]
    num_variables = pop.shape[1]

    assert num_individuals % 2 == 0

    offspring = []
    for i in range(pool_over_pop_size - 1):
        shuffled_pop = np.random.permutation(pop)
        for i in range(0, num_individuals, 2):
            x = shuffled_pop[i]
            y = shuffled_pop[i+1]

            crossover_size = np.random.randint(1, num_variables - 2)
            left = np.random.randint(0, num_variables - (crossover_size - 1))
            right = left + (crossover_size - 1)              
            # print(crossover_size)
            # print(left, right)

            x_leftover = [a for a in x if a not in y[left:right+1]]
            # print(x_leftover)
            # print(y[left:right+1])
            new_x = np.hstack((x_leftover[:left], y[left:right+1], x_leftover[left:]))
            offspring.append(new_x)

            y_leftover = [a for a in y if a not in x[left:right+1]]
            # print(y_leftover)
            # print(x[left:right+1])
            new_y = np.hstack((y_leftover[:left], x[left:right+1], y_leftover[left:]))
            offspring.append(new_y)

            # print(x)
            # print(y)
            # print(new_x)
            # print(new_y)
            # print()
    
    offspring = np.array(offspring, dtype=np.int32)
    return offspring


class Evaluation():
    def __init__(self, cost_matrix):
        self.cost_matrix = cost_matrix
    def __call__(self, tour):
        tour = np.hstack((0, tour.copy(), 0))
        # print(tour)
        # print([self.cost_matrix[tour[i], tour[i+1]] for i in range(n)])
        total_cost = sum(self.cost_matrix[tour[i], tour[i+1]] for i in range(n))
        fitness = - total_cost
        return fitness


def genetic_algorithm(cost_matrix, pop_size, verbose=0):
    num_individuals = cost_matrix.shape[0]
    assert cost_matrix.shape[1] == num_individuals
    pop = initialize_population(num_individuals, pop_size)
    algo = POPOP(variation_function=ordered_crossover, evaluation_function=Evaluation(cost_matrix), verbose=verbose)
    converged_solution = algo(pop)
    return converged_solution