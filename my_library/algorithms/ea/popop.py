from my_library.utils import CallCountWrapper

import numpy as np


class POPOP():
    def __init__(self, variation_function, evaluation_function, pool_over_pop_size=2, selection_pressure=2, verbose=0):
        self.variate = variation_function
        self.evaluate = CallCountWrapper(evaluation_function)
        self.pool_over_pop_size = pool_over_pop_size
        self.selection_pressure = selection_pressure
        self.verbose = verbose

    def num_eval_cals(self):
        return self.evaluate.num_calls

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

        new_pop = np.array(new_pop)
        new_pop_fitness = np.array(new_pop_fitness)

        return new_pop, new_pop_fitness

    def __call__(self, population, max_generation=float('+inf'), max_eval_cals=float('+inf')):
        pop = population
        pop_fitness = np.array([self.evaluate(ind) for ind in pop])
        if self.verbose:
            print('#First Population:')
            print(pop)
            print("#Gen 0:")
            print(pop_fitness)

        generation = 0
        while generation < max_generation:
            offspring = self.variate(pop, self.pool_over_pop_size)

            if self.num_eval_cals() + len(offspring) > max_eval_cals:
                break
            offspring_fitness = np.array([self.evaluate(ind) for ind in offspring])

            pool = np.vstack((pop, offspring))
            pool_fitness = np.hstack((pop_fitness, offspring_fitness))
            pop, pop_fitness = self.tournament_selection(pool, pool_fitness)

            if self.verbose:
                print(f"#Gen {generation}:")
                print(pop_fitness)
            generation += 1

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