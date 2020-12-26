import torch


class Dataset:
    def __init__(self, x, y, bs=32, shuffle=True):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.bs = bs
        self.shuffle = shuffle
        self.idx = 0

    def __iter__(self):
        if self.shuffle:
            shuffle_idx = torch.randperm(len(self.x))
            self.x, self.y = self.x[shuffle_idx], self.y[shuffle_idx]
        return self

    def __next__(self):
        if self.idx == len(self):
            raise StopIteration
        if self.idx > len(self):
            self.idx = len(self)
        x = self.x[self.idx:self.idx + self.bs]
        y = self.y[self.idx:self.idx + self.bs]
        if self.idx != len(self):
            self.idx += self.bs
        return x, y

    def __len__(self):
        return len(self.x)
