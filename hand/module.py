

class Module:
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self, inp):
        raise Exception('not implemented')

    def bwd(self, out, inp):
        raise Exception('not implemented')

    def backward(self):
        self.bwd(self.out, *self.args)
