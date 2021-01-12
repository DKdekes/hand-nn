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
        start = self.idx * self.bs
        end = start + self.bs
        if start > len(self):
            raise StopIteration
        if end > len(self):
            end = len(self)
        x = self.x[start:end]
        y = self.y[start:end]
        self.idx += 1
        return x, y

    def __len__(self):
        return len(self.x)
