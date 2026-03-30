class Data:
    """ Data() is a singular scalar value and its gradient """
    def __init__(self, data, _children=(), _op='', _label =''):
        self.data = data 
        self.grad = 0.0
        self.label = _label
        self._backpropcalc = lambda: None # function to compute gradients for this node, init to none  
        self._children = set(_children)
        self._op = _op 

    # backpropogation: sort in topo and calculate gradient for all
    def backprop(self):
        topo = []
        visited = set()
        def build_topo(v):
            if (v not in visited):
                visited.add(v)
                for child in v._children: 
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # reset gradients for all nodes in the graph before computing new gradients
        for node in topo:
            node.grad = 0.0

        self.grad = 1.0
        for node in reversed(topo):
            node._backpropcalc()


    # operation functions 
    def __add__(self, other):
        other = other if isinstance(other, Data) else Data(other)
        outval = Data(self.data + other.data, (self, other), '+')
        def _backpropcalc():
            self.grad += 1.0 * outval.grad
            other.grad+= 1.0 * outval.grad
        outval._backpropcalc = _backpropcalc # whenever you want the value, you calculate it using the function here
        return outval    

    def __mul__(self, other):
        other = other if isinstance(other, Data) else Data(other)
        outval = Data(self.data * other.data, (self, other), '*')
        def _backpropcalc():
            self.grad += outval.grad * other.data 
            other.grad += outval.grad * self.data
        outval._backpropcalc = _backpropcalc 
        return outval 
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        outval = Data(self.data**other, (self,), f'**{other}')

        def _backpropcalc():
            self.grad += (other * self.data**(other-1)) * outval.grad
        outval._backpropcalc = _backpropcalc
        return outval

    #relu 
    def __relu__(self):
        # relu(x) = x if >= 0, else 0
        outval = Data(0 if self.data < 0 else self.data, (self,), 'ReLU')
        # so derivateive = 1*out else 0
        def _backpropcalc():
            self.grad += (outval.data > 0) * outval.grad 
        outval._backpropcalc = _backpropcalc
        return outval
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1


    # representation function for when its called 
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"