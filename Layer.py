class Layer:
  
    def __init__(self, input_shape, output_shape, activation='linear'):
        self.input_length=input_shape
        self.output_length=output_shape
        self.activation=activation
        self.weight = 0
        self.bias = 0
    
    def feedForward(self, x):
          
        def activate(val):
            function = self.activation

            if function=='linear':
                return val
            
            if function=='sigmoid':
                return (1/(1+math.exp(-val)))
        
            if function=='relu':
                return np.max(0,val)
            
            if function=='tanh':
                return np.tanh(val)
            
            if function=='softmax':
                numerator = np.exp(val)
                denominator = np.sum(numerator)
                return numerator/denominator
    
        val = self.weight*x+self.bias
        return activate(val)
