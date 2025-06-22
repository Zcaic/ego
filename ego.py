import numpy as np 
from smt.applications import EGO 
from smt.surrogate_models import KRG 

class MEGO(EGO):
    def optimize(self, fun):
        return super().optimize(fun)

KRG().nx