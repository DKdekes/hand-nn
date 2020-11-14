from hand.module import Module


class Linear(Module):
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, inp):
        print(f'{__name__} forward')
        return inp @ self.w + self.b

    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g
        self.b.g = out.g.sum()
