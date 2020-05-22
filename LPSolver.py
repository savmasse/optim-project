
import scipy.optimize as lp
import numpy as np

"""
Class for the solving of standard linear programs.
"""
class LPSolver():
    
    def __init__(self, 
                 ih, 
                 SG=1.050, 
                 y=None,
                 method=None, 
                 tol=10**(-8)):
        
        self.ih = ih
        ih.y = y
        self.SG = SG
        self.c = []
        self.c0 = 0
        self.A_ub = []
        self.b_ub = []
        self.A_eq = []
        self.b_eq = []
        self.bounds = []
        self.y = y
        self.V = None
        
        # Set the solving method for linprog
        if method is not None:
            self.method = method
        else:
            self.method = "interior-point"
        self.tol = tol
        
    def add_quality(self, q_min, i="mout"):
        """
        Add a constraint for the ingredient quality.
        """
        a_ub = None
        if i == 'mout':
            a_ub = self.ih.avg_q_mout(q_min)
        elif i == "hop":
            a_ub = self.ih.avg_q_hop(q_min)
        elif i == "gist":
            a_ub = self.ih.avg_q_gist(q_min)
        else:
            raise Exception("Invalid ingredient specified.")
            
        # Add to the list of constraints
        self.A_ub.append(-a_ub) # Negative to change from lower to upper bound
        self.b_ub.append(0)
        
    
    def add_ebc_lower(self, ebc):
        """
        Get a constraint for the EBC lower bound.
        """
        # Add the lower bound
        vec_lower = self.ih.ebc_vector(ebc)
        self.A_ub.append(-vec_lower)
        self.b_ub.append(0)
        
    def add_ebc_upper(self, ebc):
        """
        Get a constraint for the EBC upper bound.
        """
        # Add the upper bound
        vec_upper = self.ih.ebc_vector(ebc)
        self.A_ub.append(vec_upper)
        self.b_ub.append(0)
    
    def add_cost(self):
        """
        Add a constraint for the total cost of ingredient and processes.
        """
        c, c0 = self.ih.total_cost_vector()
        self.c0 = c0 # Set the constant if there is one
        self.c = c # Set the objective function
    
    def add_volume(self, vol_req):
        """
        Add a constraint equation for the volume.
        """
        # Set the volume for later use
        self.V = vol_req
        # Calculate the constraint vector for the volume constraint
        vec = self.ih.total_extract_equivalent() #* 0.95 / (2.9 * (self.SG-1))
        
        # Add to the constraint list
        self.A_eq.append(vec)
        self.b_eq.append(vol_req * 2.9 * (self.SG-1) / 0.95)

    def add_hop_req(self):
        """
        Add the constraints for the hop weights based on the beer volume.
        """
        # Add to constraint list
        vec, c = self.ih.hop_vector(self.V)
        self.A_eq.append(vec)
        self.b_eq.append(c)
        
        
    def add_gist_req(self):
        """
        Add the constraints for the gist weigths based on the beer volume.
        """
        # Add to constraint list
        vec, c = self.ih.gist_vector(self.V)
        self.A_eq.append(vec)
        self.b_eq.append(c)
        
    def reset(self):
        """
        Reset the solver so a new problem can be added.
        """
        self.A_eq = []
        self.b_eq = []
        self.A_ub = []
        self.b_ub = []
        self.c = []
        self.bounds = []
        
    def set_bounds(self):
        """
        Set the bounds for all the decision variables.
        """
        b = [(0, None) for _ in range(self.ih.dim - 2)]
        if self.y is None:
            # In the relaxed problem there are two extra variables with bounds [0;1]
            b.append((0, 1))
            b.append((0, 1))
        self.bounds = b
        
    def add_summation_constraints(self, M=1_000_000):
        """
        When solving through summation two extra constraints must be added
        to force x=0 if y=0.
        """
        if self.y is None:
            return
        
        v1, v2 = self.ih.summation_vectors()
        # Add first constraint
        self.A_ub.append(v1)
        self.b_ub.append(M*self.y[0])
        
        # Add second constraint
        self.A_ub.append(v2)
        self.b_ub.append(M*self.y[1])
        
        
    def add_relaxed_constraint(self, M=1_000_000):
        """
        Add the relaxed constraints for the fixed production costs.
        """
        v1, v2 = self.ih.summation_vectors()
        v1[-2] = -M
        v2[-1] = -M
        
        # Add first constraint
        self.A_ub.append(v1)
        self.b_ub.append(0)
        
        # Add second constraint
        self.A_ub.append(v2)
        self.b_ub.append(0)

        
    def _solve(self,
               vol,
               ebc_min=None,
               ebc_max=None,
               q_mout=None,
               q_hop=None,
               q_gist=None,
               M=10000,
               display=False):
        """
        Solve the linear program with the given constraint. Can be solved through
        summation or in a relaxed form.
        """
        
        # Add the constraints that are shared between all problems
        self.add_volume(vol)
        self.add_hop_req()
        self.add_gist_req()
        self.add_cost()
        
        # Add quality if requested
        if q_mout is not None:
            self.add_quality(q_mout, "mout")
        if q_hop is not None:
            self.add_quality(q_hop, "hop")
        if q_gist is not None:
            self.add_quality(q_gist, "gist")
            
        # Add EBC if requested
        if ebc_min is not None:
            self.add_ebc_lower(ebc_min)
        if ebc_max is not None:
            self.add_ebc_upper(ebc_max)
            
        # If solving through summation, then add additional constraints
        if self.y is not None:
            self.add_summation_constraints(M)
        else:
            self.add_relaxed_constraint(M)
            
        # Set bounds for the decision variables
        self.set_bounds()
            
        # Now solve the problem and 
        res = lp.linprog(method=self.method,
                          c=self.c,
                          A_ub=self.A_ub if len(self.A_ub) else None, 
                          b_ub=self.b_ub if len(self.b_ub) else None, 
                          A_eq=self.A_eq,
                          b_eq=self.b_eq,
                          bounds=self.bounds,
                          options={"disp": display,"tol":self.tol})
        fun = res.fun
        
        # Summation solution does not include the fixed cost values, so add them
        if self.y is not None and self.y[0] == 1:
            fun += self.ih.mouten.fixed_cost
        if self.y is not None and self.y[1] == 1:
            fun += self.ih.maischen.fixed_cost
        
        return res, fun
        

    def solve(self, 
              vol,
              summation=False,
              ebc_min=None,
              ebc_max=None,
              q_mout=None,
              q_hop=None,
              q_gist=None,
              M=10000,
              display=False):
        
        # Return relaxation if no summation required
        if not summation:
            return self._solve(vol, 
                               ebc_min=ebc_min,
                               ebc_max=ebc_max,
                               q_mout=q_mout,
                               q_hop=q_hop,
                               q_gist=q_gist,
                               M=M,
                               display=display)
        
        # If summation, then iterate over all possible binary combinations
        else:
            l = [(0, 0), (0, 1), (1, 0), (1, 1)]
            rlist = []
            flist = []
            
            for y in l:
                self.y = y
                self.ih.y = y
            
                res, f = self._solve(vol, 
                                   ebc_min=ebc_min,
                                   ebc_max=ebc_max,
                                   q_mout=q_mout,
                                   q_hop=q_hop,
                                   q_gist=q_gist,
                                   M=M,
                                   display=display)
                rlist.append(res)
                flist.append(f)
                
                self.reset()
            
            # Remove those that were unsuccesful
            failed = [not r.success for r in rlist]
            if np.all(failed):
                raise Exception("All failed.")
            
            flist = np.array(flist)
            flist[failed] = np.inf
            best_index = np.argmin(np.array(flist))
            
            return rlist, flist, best_index
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    