from neuron.core.neuron import Neuron
from neuron.core.params_loader import params


class Synapse:
    CSYN= params['Csyn']
    TS = params['Ts']

    def __init__(self, w=0.5, pre: 'Neuron' = None, post: 'Neuron' = None) -> None:
        self.connect(pre, post)
        self.set_weight(w)
        self.Isyn = 0


    def connect(self, pre: 'Neuron' = None, post: 'Neuron' = None):
        if pre:
            self.pre_neuron = pre
        
        if post:
            self.post_neuron = post

    def set_weight(self, w):
        self.w = w

    def step(self, t, dt):
        if self.pre_neuron.s:
            self.post_neuron.charge(Synapse.CSYN*self.w)

    def __str__(self) -> str:
        return f"Synapse from: {self.pre_neuron} to: {self.post_neuron}"