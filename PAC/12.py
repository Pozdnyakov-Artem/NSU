import numpy as np


class Neuron():
    def __init__(self, n_inputs=2, ler_rat=0.1):
        self.weight = np.random.random(n_inputs)
        self.bias = np.random.random()
        self.ler_rat = ler_rat

    def forward(self, inp):
        d = np.dot(self.weight, inp) + self.bias
        return 1/(1+np.exp(-d))

    def MSE(self, out, true):
        return ((out - true)**2).mean()

    def grad(self, inp, out, true, loss):
        inp = np.array(inp)
        pr_sig = out * (1-out)

        local = loss * pr_sig

        de_dw = self.ler_rat* local * inp
        de_db = self.ler_rat*local

        return de_dw, de_db, local

    def backward(self, inp, out, true, loss):
        dw, db, local = self.grad(inp, out, true, loss)
        self.weight = self.weight - dw
        self.bias = self.bias - db

        return local


class Model():
    def __init__(self):
        self.neurons = [Neuron(), Neuron(), Neuron()]

    def forward(self, inp):
        n1 = self.neurons[0].forward(inp)
        n2 = self.neurons[1].forward(inp)

        hid_out = [n1, n2]

        out = self.neurons[2].forward(hid_out)

        return out, hid_out

    def backward(self, inp, hid_out, out, true, loss):
        grad = self.neurons[2].backward(hid_out, out, true, loss)
        self.neurons[1].backward(inp, hid_out[1], true, self.neurons[2].weight[1] * grad)
        self.neurons[0].backward(inp, hid_out[0], true, self.neurons[2].weight[0] * grad)

    def fit(self,epochs = 1000):
        data = {(0,1):1, (1,0):1, (1,1):0, (0,0):0}

        for epoch in range(epochs):
            for inp, true in data.items():
                out, hid_out = self.forward(inp)
                # loss = (out - true)
                loss = (out - true) / (out * (1 - out))
                self.backward(inp, hid_out, out, true, loss)





model = Model()
model.fit(10000)
print(model.forward([0,0])[0])
print(model.forward([1
                        ,1])[0])
print(model.forward([1,0])[0])
print(model.forward([0,1])[0])