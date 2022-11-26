import numpy as np
from itertools import product

class zip2():
    def __init__(self, *args, random_order=False):
        self.args = args
        self.random_order = random_order

        self.len = 1
        self.lengths = []
        self.tuple_lst_idx = []
        for it in args:
            self.lengths.append(len(it))
            self.len *= len(it)
            self.tuple_lst_idx.append(np.arange(len(it), dtype=int))
        self.lengths = tuple(self.lengths)
        
        self.tuple_lst_idx = np.array(list(product(*self.tuple_lst_idx)), dtype=int)
        self.rand_int_idx = np.arange(self.len)
        np.random.shuffle(self.rand_int_idx)


    def __len__(self):
        return self.len

    def __iter__(self):
        self.iter_idx = -1
        if self.random_order:
            np.random.shuffle(self.rand_int_idx)

        return self

    def __next__(self):
        self.iter_idx += 1
        if self.iter_idx >= self.len:
            raise StopIteration
        if self.random_order:
            idx = self.rand_int_idx[self.iter_idx]
        else:
            idx = self.iter_idx
            
        return self.get_item_from_int(idx)

    def __getitem__(self, key):
        try:
            return self.get_item_from_int(key)
        except:
            try:
                return self.get_item_from_tuple(key)
            except:
                try:
                    return self.get_item_from_list(key)
                except:
                    raise TypeError('Indices must be a integer, a list or a tuple')

    def get_item_from_list(self, key):
        if not isinstance(key, list):
            raise TypeError('Indices must be a list')
        res = []
        for idx in key:
            try:
                res.append(self.get_item_from_int(idx))
            except:
                try:
                    res.append(self.get_item_from_tuple(idx))
                except:
                    raise TypeError('A index must be a integer or a tuple')

        return res

    def get_item_from_int(self, key):
        if not isinstance(key, int):
            raise TypeError('Index must be a integer')

        tuple_idx = tuple(self.tuple_lst_idx[key])
        return self.get_item_from_tuple(tuple_idx)

    def get_item_from_tuple(self, key):
        if not isinstance(key, tuple):
            raise TypeError('Index must be a tuple')
        index_error = False
        index_error = index_error or (len(key) != len(self.args))
        for i in range(len(self.args)):
            index_error = index_error or (key[i] >= self.lengths[i])

        if index_error:
            raise IndexError('Index out of range')

        res = tuple(it[key[i]] for i, it in enumerate(self.args))
        return res