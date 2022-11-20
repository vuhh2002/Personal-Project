from my_library.utils import (
    normal_distribution,
    CallCountWrapper,
    tuple_all_sublist,
)

import numpy as np


class CEM:
    def __init__(
        self,
        num_params,
        mu_init = None,
        cov_init = 1e-3,
        epsilon = 1e-3,
        epsilon_limit = 1e-5,
        decay = 0.95,
        ):

        self.num_params = num_params
        if mu_init != None:
            self.mu = np.array(mu_init)
        else:
            self.mu = np.zeros(self.num_params)
        self.cov = np.array(cov_init)
        if len(self.cov) == 1:
            self.cov = cov_init * np.ones(self.num_params)

        self.epsilon = epsilon
        self.epsilon_limit = epsilon_limit
        self.decay = decay

        self.best_ind = None
        self.best_fitness = None
        self.num_parents = None
        self.weights = None

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        
        pop = normal_distribution(pop_size, self.mu, self.cov)
        return pop

    def tell(self, population, fitness, num_parents=None, elitism=True):
        """
        Updates the distribution
        """

        if num_parents == None:
            num_parents = len(population) / 2
        if self.num_parents != num_parents:
            self.num_parents = num_parents
            self.weights = np.array([np.log((num_parents + 1) / i)
                                    for i in range(1, num_parents + 1)])
            self.weights /= self.weights.sum()
            
        if self.best_ind != None and elitism:
            population = np.hstack((self.best_ind, population))
            fitness = np.hstack((self.best_fitness, fitness))
        fitness = np.array(fitness)
        idx_sorted = np.argsort(- fitness)
        self.best_ind = population[idx_sorted[0]]
        self.best_fitness = fitness[idx_sorted[0]]

        old_mu = self.mu
        self.epsilon = self.epsilon * self.decay + (1 - self.decay) * self.epsilon_limit

        self.mu = self.weights @ population[idx_sorted[:self.num_parents]]
        diff = population[idx_sorted[:self.num_parents]] - old_mu
        self.cov = self.weights @ (diff**2) + self.epsilon * np.ones(self.num_params)


def solve(
    fobj,
    num_params,
    pop_size,
    max_eval_cals,
    max_generation = float('+inf'),
    num_parents = None,
    elitism = True,
    mu_init = None,
    cov_init = 1e-3,
    epsilon = 1e-3,
    epsilon_limit = 1e-5,
    decay = 0.95,
    log_mode = 0, # 0, 1, 2 or 3
    verbose = False,
    ):

    fobj = CallCountWrapper(fobj)
    cem = CEM(
        num_params,
        mu_init,
        cov_init,
        epsilon,
        epsilon_limit,
        decay,
    )

    results = []
    all_pops = []
    generation = 0
    while generation < max_generation and fobj.num_calls + pop_size <= max_eval_cals:
        pop = cem.ask(pop_size)
        fitness = [fobj(ind) for ind in pop]
        cem.tell(pop, fitness, num_parents, elitism)

        if log_mode == 1 or log_mode == 3:
            row = (generation, fobj.num_calls, cem.best_ind, cem.best_fitness)
            results.append(row)
        if log_mode == 2 or log_mode == 3:
            all_pops.append(pop)
        if verbose:
            print(f'Generation {generation}, best fitness: {cem.best_fitness}')
        generation += 1

    results = tuple_all_sublist(results)
    all_pops = tuple_all_sublist(all_pops)
    if log_mode == 0:
        return cem.best_ind, cem.best_fitness
    if log_mode == 1:
        return results
    if log_mode == 2:
        return all_pops
    if log_mode == 3:
        return results, all_pops
    
    