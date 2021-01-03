import math
import numpy as np

class LinearNN(SingleLayerNN):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def addLayer(self):
        self.mainLayer=Layer(1, 1)

    def train(self, epochs=300, learning_rate=0.005):

        def linearDerivatives():
            bD = 0
            wD = 0
            for inp, out in zip(self.x, self.y):
                predicted = self.mainLayer.feedForward(inp)
                bias = (predicted-out)
                weight = (predicted-out)*inp
                bD+=bias
                wD+=weight
            bD = bD/len(self.x)
            wD = wD/len(self.x)
            return bD, wD
        
        for i in range(epochs)
            print("Epochs="+str(epochs))
            print("Weight:",self.mainLayer.weight)
            print("Bias:",self.mainLayer.bias)
            print('Loss:',self.cost())
            print('\n')

            bd, wd = linearDerivatives()
            
            self.mainLayer.weight=self.mainLayer.weight-learning_rate*wd
            self.mainLayer.bias=self.mainLayer.bias-learning_rate*bd
        
    def predict(self, xval):
        return self.mainLayer.feedForward()
    
    def cost(self):
        cost = 0
        for inp, out in zip(self.x, self.y):
            predicted = self.mainLayer.feedForward(inp)
            loss = (predicted-out)**2
            cost+=loss
        cost = cost/(len(self.x)*2)
        return cost
