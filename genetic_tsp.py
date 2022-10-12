import numpy as np


class POPOP():
    def __init__(self, variation_function, evaluation_function, pool_over_pop_size=2, selection_pressure=2, verbose=0):
        self.variate = variation_function
        self.evaluation_function = evaluation_function
        self.pool_over_pop_size = pool_over_pop_size
        self.selection_pressure = selection_pressure
        self.verbose = verbose
        self.n_eval_call = 0

    def evaluate(self, ind):
        self.n_eval_call += 1
        return self.evaluation_function(ind)

    def tournament_selection(self, pool, pool_fitness):
        pool_size = pool.shape[0]
        group_size = self.pool_over_pop_size * self.selection_pressure
        assert pool_size >= group_size and pool_size % group_size == 0

        new_pop = []
        new_pop_fitness = []

        for i in range(self.selection_pressure):
            shuffled_idx = np.random.permutation(pool_size)

            for j in range(0, pool_size, group_size):
                group_idx = shuffled_idx[j:j + group_size]
                group_fitness = pool_fitness[group_idx]
                winner_idx = group_idx[group_fitness.argmax()]
                new_pop.append(pool[winner_idx])
                new_pop_fitness.append(pool_fitness[winner_idx])

                # print(pool[group_idx])
                # print(group_fitness)

        new_pop = np.array(new_pop)
        new_pop_fitness = np.array(new_pop_fitness)
        # print()
        # print(new_pop)
        # print(new_pop_fitness)
        # print('####################')

        return new_pop, new_pop_fitness

    def __call__(self, population, num_generations=float('+inf')):
        pop = population
        pop_fitness = np.array([self.evaluate(ind) for ind in pop])
        if self.verbose:
            print('#First Population:')
            print(pop)
            print("#Gen 0:")
            print(pop_fitness)

        generation = 0
        while generation < num_generations:
            offspring = self.variate(pop, self.pool_over_pop_size)
            offspring_fitness = np.array([self.evaluate(ind) for ind in offspring])
            pool = np.vstack((pop, offspring))
            pool_fitness = np.hstack((pop_fitness, offspring_fitness))
            pop, pop_fitness = self.tournament_selection(pool, pool_fitness)
            # print()
            # print(pool)
            # print(pop)
            # print(pool_fitness)
            # print(pop_fitness)
            # print("#############")

            generation += 1
            if self.verbose:
                print(f"#Gen {generation}:")
                print(pop_fitness)

            if np.all(pop == pop[0]):
                if self.verbose:
                    print("Population converged")
                self.converged_generation = generation
                break

        best_idx = pop_fitness.argmax()
        self.best_ind = pop[best_idx]
        self.best_fitness = pop_fitness[best_idx]
        if self.verbose:
            print('#Best Individual:')
            print(self.best_ind)

        return self.best_ind


def initialize_population(n, pop_size):
    pop = []
    for i in range(pop_size):
        ind = np.arange(n)
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
            left = np.random.randint(1, num_variables - (crossover_size - 1))
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
        tour = np.hstack((tour.copy(), 0))
        # print(tour)
        # print([self.cost_matrix[tour[i], tour[i+1]] for i in range(n)])
        total_cost = sum(self.cost_matrix[tour[i], tour[i+1]] for i in range(n))
        return total_cost


def genetic_algorithm(cost_matrix, pop_size, verbose=0):
    num_individuals = cost_matrix.shape[0]
    assert cost_matrix.shape[1] == num_individuals
    pop = initialize_population(num_individuals, pop_size)
    algo = POPOP(variation_function=ordered_crossover, evaluation_function=Evaluation(cost_matrix), verbose=verbose)
    converged_solution = algo(pop)
    return converged_solution