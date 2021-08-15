from neuron.optimizer.neat.gene import Gene
from multipledispatch import dispatch

class ConnectionGene(Gene):
    @dispatch(int,int,int,float,bool)
    def __init__(self, innovation_number,f,t,weight,enabled) -> None:
        super().__init__(innovation_number)
        self.f = f
        self.t = t
        self.weight = weight
        self.enabled = enabled

    def get_from(self) -> int:
        return self.f
    
    def get_to(self) -> int:
        return self.t
    
    @dispatch(object)
    def __init__(self, other: 'ConnectionGene') -> None:
        super().__init__(other)
        self.f = other.f
        self.t = other.t
        self.weight = other.weight
        self.enabled = other.enabled

    def clone(self) -> 'ConnectionGene':
        return ConnectionGene(self)
    
    def __str__(self) -> str:
        return f"ConnectionGene {{innovation_number = {self.innovation_number},from={self.f},to={self.t},weight = {self.weight},enabled = {self.enabled}}}"
