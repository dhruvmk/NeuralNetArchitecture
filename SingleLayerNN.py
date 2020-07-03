from abc import abstractmethod, ABC


class SingleLayerNN(ABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @abstractmethod
    def addLayer(self):
        pass

    def cost(self):
        cost = 0
        for inp, out in zip(self.x, self.y):
            predicted = self.mainLayer.feedForward(inp)
            loss = (predicted-out)**2
            cost+=loss
        cost = cost/(len(self.x)*2)
        return cost

    def predict(self, xval):
        return self.mainLayer.feedForward()
