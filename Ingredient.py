class Ingredient():
    """
    Class to store ingredient data; every ingredient should know its own name.
    """
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.dim = data.shape[0]
        
    def vec(self, col=None):
        return self.data[col].to_numpy()
    
    def cost(self):
        return self.vec("kost")
        
    def q(self):
        return self.vec("waardering")
    
    def ebc(self):
        if hasattr(self.data, "EBC"):
            return self.vec("EBC")
        else:
            return None