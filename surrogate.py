import casadi as ca 
import aerosandbox.numpy as anp 

class Surrogate():
    def __init__(self):
         pass 
    def predict(self):
        pass 
    def predict_derivate(self,order):
        pass 


class CallbackSurrogate(ca.Callback):
    def __init__(self,name,surrogate:Surrogate, opts={}):
        self.surrogate=surrogate
        self.construct(name,opts)