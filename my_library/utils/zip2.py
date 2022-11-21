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
        idx = [0] * len(self.args)
        for i in range(len(self.args) - 1, -1, -1):
            idx[i] = key % self.lengths[i]
            key = key // self.lengths[i]

        return self.get_item_from_tuple(tuple(idx))

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