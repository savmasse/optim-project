import numpy as np
from Ingredient import Ingredient
from Process import Process
        
class IngredientHelper():
    """
    Contains helper functions to easily access ingredient data and process data.
    """
    
    def __init__(self,
                 gerst_data,
                 mout_data,
                 extract_data,
                 hop_data,
                 gist_data,
                 mout_process,
                 maisch_process,
                ):
        
        self.gerst = Ingredient("gerst", gerst_data)
        self.mout = Ingredient("mout", mout_data)
        self.extract = Ingredient("extract", extract_data)
        self.hop = Ingredient("hop", hop_data)
        self.gist = Ingredient("gist", gist_data)
        self.mouten = Process("mouten", mout_process)
        self.maischen = Process("maischen", maisch_process)
        self.y = None
        
        # 2 binary variables added
        self.dim = 2 + sum([self.gerst.dim, self.mout.dim, self.extract.dim, self.hop.dim, self.gist.dim])
        
    def get_index(self, i):
        """
        Get the correct insert indices for each vector
        """
        l = 0
        if i == "gerst":
            return np.arange(l, self.gerst.dim)
        l += self.gerst.dim
        if i == "mout":
            return np.arange(l, l + self.mout.dim)
        l += self.mout.dim
        if i == "extract":
            return np.arange(l, l + self.extract.dim)
        l += self.extract.dim
        if i == "hop":
            return np.arange(l, l + self.hop.dim)
        l += self.hop.dim
        if i == "gist":
            return np.arange(l, l + self.gist.dim)
        
    def _empty_vec(self):
        """
        Return a vector of zeros of the appropriate length depending on the solving method.
        """
        if self.y is None:
            return np.zeros(self.dim)
        else:
            return np.zeros(self.dim-2)
        
    def total_extract_equivalent(self):
        """
        Returns a vector of the mout sources in moutextract equivalent.
        """
        vec = self._empty_vec()
        vec[self.get_index("gerst")] = 0.75*0.8
        vec[self.get_index("mout")] = 0.8
        vec[self.get_index("extract")] = 1.0
        return vec
    
    def avg_q_hop(self, Q_hop):
        """
        Returns vector for the average quality of hops.
        """
        vec = self._empty_vec()
        vec[self.get_index("hop")] = (self.hop.q() - Q_hop)
        return vec
    
    def avg_q_gist(self, Q_gist):
        """
        Returns vector for the average quality of gist.
        """
        vec = self._empty_vec()
        vec[self.get_index("gist")] = (self.gist.q() - Q_gist)
        return vec
    
    def avg_q_mout(self, Q_mout):
        """
        Returns vector for the average quality of different sources of mout.
        """
        vec = self.total_extract_equivalent()
        vec[self.get_index("gerst")] *= (self.gerst.q() - Q_mout)
        vec[self.get_index("mout")] *= (self.mout.q() - Q_mout)
        vec[self.get_index("extract")] *= (self.extract.q() - Q_mout)
        return vec
        
    def _ingredient_cost_vector(self):
        """
        Returns the partial constraint vector for the cost of the ingredients.
        """
        vec = self.gerst.cost()
        vec = np.append(vec, self.mout.cost())
        vec = np.append(vec, self.extract.cost())
        vec = np.append(vec, self.hop.cost())
        vec = np.append(vec, self.gist.cost())
        if self.y is None:
            vec = np.append(vec, np.array([0]*2)) # Add two zeros for the y variables
        return vec
    
    def _mouten_cost_vector(self):
        """
        Returns the partial constraint vector for the cost of the mouten process.
        """
        vec = self._empty_vec()
        c0 = 0
        if self.y is not None and self.y[0] == 1:
            c0 = self.mouten.fixed_cost
        if self.y is None:
            vec[-2] = self.mouten.fixed_cost
        vec[self.get_index("gerst")] = self.mouten.var_cost*0.75
        
        return (vec, c0)
    
    def _maischen_cost_vector(self):
        """
        Returns the partial constraint vector for the cost of the mouten process.
        """
        vec = self._empty_vec()
        c0 = 0
        if self.y is not None and self.y[1] == 1:
            c0 = self.maischen.fixed_cost
        if self.y is None:
            vec[-1] = self.maischen.fixed_cost
            
        vec[self.get_index("gerst")] = 0.75*self.maischen.var_cost*0.8
        vec[self.get_index("mout")] = self.maischen.var_cost*0.8
        
        return (vec, c0)
    
    def total_cost_vector(self):
        """
        Return the vector for the total cost of the ingredients and the processes.
        """
        vec = self._ingredient_cost_vector() + self._mouten_cost_vector()[0] + self._maischen_cost_vector()[0]
        c0 = self._mouten_cost_vector()[1] + self._maischen_cost_vector()[1]
        return (vec, c0)
    
    
    def ebc_vector(self, ebc_bound):
        """
        Return the vector for the EBC constraints.
        """
        vec = self.total_extract_equivalent()
        vec[self.get_index("gerst")] *= (self.gerst.ebc() - ebc_bound)
        vec[self.get_index("mout")] *= (self.mout.ebc() - ebc_bound)
        vec[self.get_index("extract")] *= (self.extract.ebc() - ebc_bound)
        return vec
    
    
    def summation_vectors(self):
        """
        Return vectors for binary summation constraints.
        """
        vec_y1 = self._empty_vec()
        vec_y1[self.get_index("gerst")] = 1
        
        vec_y2 = self._empty_vec()
        vec_y2[self.get_index("gerst")] = 0.75 # This is after moutproces
        vec_y2[self.get_index("mout")] = 1
        
        return vec_y1, vec_y2
    
    def hop_vector(self, V, SG=1.050):
        c = 0.0013*V/0.95
        vec = self._empty_vec()
        vec[self.get_index("hop")] = 1
        return vec, c
        
    def gist_vector(self, V, SG=1.050):
        c = 0.075*V/100
        vec = self._empty_vec()
        vec[self.get_index("gist")] = 1
        return vec, c
        