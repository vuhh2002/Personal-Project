class zip2():
    def __init__(self, *args):
        self.args = args

        self.len = 1
        self.lengths = []
        for it in args:
            self.lengths.append(len(it))
            self.len *= len(it)
        self.lengths = tuple(self.lengths)
        # print(self.lengths)
        # print(self.len)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.next_idx = 0
        return self

    def __next__(self):
        if self.next_idx >= self.len:
            raise StopIteration

        res = self[self.next_idx]
        self.next_idx += 1
        
        return res

    def __getitem__(self, indices):
        try:
            res = tuple(self.get_item_from_int_idx(i) for i in indices)
        except:
            if not isinstance(indices, int):
                raise TypeError('Indices must be integers or slices')
            res = self.get_item_from_int_idx(indices)

        return res

    def get_item_from_int_idx(self, index):
        idx = [0] * len(self.args)
        for i in range(len(self.args) - 1, -1, -1):
            idx[i] = index % self.lengths[i]
            index = index // self.lengths[i]

        return self.get_item_from_sparated_idx(idx)

    def get_item_from_sparated_idx(self, indice):
        index_error = False
        index_error = index_error or (len(indice) != len(self.args))
        for i in range(len(self.args)):
            index_error = index_error or (indice[i] >= self.lengths[i])

        if index_error:
            raise IndexError('list assignment index out of range')

        res = tuple(it[indice[i]] for i, it in enumerate(self.args))
        return res