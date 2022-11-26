from library.utils import (
    normal_distribution,
    CallCountWrapper,
)

import numpy as np


class CEM:
    def __init__(
        self,
        num_params,
        bounds = [float('-inf'), float('+inf')],
        range_init = [-1, 1],
        elitism = True,
        ):

        self.num_params = num_params

        bounds = np.array(bounds)
        if bounds.shape == (2,):
            bounds = np.ones((num_params, 2)) * bounds
        self.lower_bound = bounds[:, 0]
        self.upper_bound = bounds[:, 1]

        range_init = np.array(range_init)
        if range_init.shape == (2,):
            range_init = np.ones((num_params, 2)) * range_init
        lower_range_init = range_init[:, 0]
        upper_range_init = range_init[:, 1]
        diff = upper_range_init - lower_range_init
        self.mu = lower_range_init + diff / 2
        self.cov = (diff / 2) ** 2

        self.elitism = elitism
        self.best_ind = None
        self.best_loss = None
        self.num_parents = None
        self.weights = None

        self.c_1 = 2 / num_params**2
        self.c_c = 4 / num_params
        self.p_c = np.zeros(num_params)

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """

        pop = normal_distribution(pop_size, self.mu, self.cov)
        pop = np.clip(pop, self.lower_bound, self.upper_bound)

        return pop

    def tell(self, pop, loss, num_parents=None):
        """
        Updates the distribution
        """

        if num_parents == None:
            num_parents = len(pop) // 2
        if self.num_parents != num_parents:
            self.num_parents = num_parents
            self.weights = np.array([np.log((self.num_parents + 1) / i)
                            for i in range(1, self.num_parents + 1)])
            self.weights /= self.weights.sum()
            self.parents_eff = 1 / (self.weights ** 2).sum()
            
        if self.best_ind is not None and self.elitism:
            pop = np.vstack((self.best_ind, pop))
            loss = np.hstack((self.best_loss, loss))
        loss = np.array(loss)
        idx_sorted = np.argsort(loss)
        self.best_ind = pop[idx_sorted[0]]
        self.best_loss = loss[idx_sorted[0]]

        prev_mu = self.mu
        self.mu = self.weights @ pop[idx_sorted[:self.num_parents]]
        self.p_c = (1 - self.c_c) * self.p_c + np.sqrt(self.c_c * (2 - self.c_c) * self.parents_eff) * (self.mu - prev_mu)
        self.cov = (1 - self.c_1) * self.cov + self.c_1 * self.p_c * self.p_c.T


def solve(
    fobj,
    num_params,
    pop_size,
    max_eval_cals,
    bounds = [float('-inf'), float('+inf')],
    range_init = [-1, 1],
    max_generation = float('+inf'),
    num_parents = None,
    elitism = True,
    log_mode = 0, # 0, 1, 2 or 3
    verbose = False,
    ):

    fobj = CallCountWrapper(fobj)
    cem = CEM(
        num_params=num_params,
        bounds=bounds,
        range_init=range_init,
        elitism=elitism,
    )
    results = []
    all_pops = []
    generation = 0

    while generation < max_generation and fobj.num_calls + pop_size <= max_eval_cals:
        pop = cem.ask(pop_size=pop_size)
        loss = [fobj(ind) for ind in pop]
        cem.tell(pop=pop, loss=loss, num_parents=num_parents)

        if log_mode == 1 or log_mode == 3:
            row = (cem.best_ind, cem.best_loss, generation, fobj.num_calls)
            results.append(row)
        if log_mode == 2 or log_mode == 3:
            all_pops.append(pop.copy())
        if verbose:
            print(f'Generation {generation}, best loss: {cem.best_loss}')
        generation += 1

    if log_mode != 0:
        return results, all_pops
    return cem.best_ind, cem.best_loss
    
    