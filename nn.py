import random 
from engine import Data


class Neuron:
    """ A singular Neuron(num_inputs, nonlin=False), like Neuron(3) """
    def __init__(self, num_inputs, nonlin=False):
        self.nonlin=nonlin
        self.weights = [Data(random.uniform(-1, 1)) for _ in range(num_inputs)] # random initialisation
        self.bias = Data(0) # initialised to 0

    def params(self):
        return self.weights + [self.bias]
    
    def __call__(self, x_list):
        activation = sum((wi*xi for wi,xi in zip(self.weights, x_list)), self.bias)
        return activation.__relu__() if self.nonlin else activation
        # neuron = relu(x1w1 + x2w2 + ... + b)
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.weights)} weights)"

class Layer:
    """ Layer(num_inputs, num_outputs), like Layer(2, 3) """
    def __init__(self, num_inputs, num_outputs, nonlin=False):
        self.nonlin = nonlin
        self.neurons = [Neuron(num_inputs, self.nonlin) for _ in range(num_outputs)]

    def params(self):
        return [p for n in self.neurons for p in n.params()]
    
    
    def __call__(self, x):
        layerout = [neuron(x) for neuron in self.neurons]
        return layerout[0] if len(layerout)==1 else layerout
        # layer = list of neurons

    def __repr__(self):
        return f"Layer of Neurons [{', '.join(str(n) for n in self.neurons)}]"

class MLP:
    """ MLP(num_inputs, num_outputs_list), like MLP(1, [2, 1]) """
    def __init__(self, num_inputs, num_outputs_list, nonlin=False):
        self.nonlin = nonlin
        combined_nums = [num_inputs] + num_outputs_list
        self.layers = [Layer(combined_nums[i], combined_nums[i+1], self.nonlin) for i in range(len(num_outputs_list))]
        # mlp = list of layers

    def params(self):
        res = []
        for layer in self.layers:
            for p in layer.params():
                res.append(p)
        return res

    # connect them together
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP of Layers [{', '.join(str(layer) for layer in self.layers)}]"
