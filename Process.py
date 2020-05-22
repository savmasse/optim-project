
class Process():
    """
    Class to store process data.
    """
    def __init__(self, name, data):
        self.name = name
        self.fixed_cost = data["vastekost"]
        self.var_cost = data["variabelekost"]