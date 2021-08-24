from neuron.core.neuron import Neuron,params
from neuron.utils.units import ms

class Synapse:
    CSYN= params['Csyn']
    TS = params['Ts']
    def __init__(self,w=0.5,pre:'Neuron'= None,post:'Neuron'= None) -> None:
        self.connect(pre,post)
        self.set_weight(w)
        self.Isyn = 0

    
    def connect(self,pre:'Neuron'=None,post:'Neuron'=None):
        if pre:
            self.pre_neuron = pre
        
        if post:
            self.post_neuron = post
            
        
    def set_weight(self, w):
        self.w = w

    def step(self,t,dt):
        self.Isyn += (dt/Synapse.TS)*(-self.Isyn) + Synapse.CSYN * (1 if self.pre_neuron.s else 0)
        if self.Isyn >= 2:
            self.Isyn = 2
        self.post_neuron.charge(self.Isyn*self.w)