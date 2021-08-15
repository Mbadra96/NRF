from neuron.optimizer.neat.gene import Gene
from enum import Enum
from multipledispatch import dispatch

class NodeType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3

class NodeGene(Gene):
    
    @dispatch(int,object)
    def __init__(self, innovation_number,type:'NodeType') -> None:
        super().__init__(innovation_number)
        if isinstance(type, NodeType):
            self.type = type
        else:
            raise TypeError(f"invaled object excepted GeneType but {type.__class__.__name__} was given instead")


    @dispatch(object)
    def __init__(self, other: 'NodeGene') -> None:
        super().__init__(other)
        self.type = other.type

    def clone(self) -> 'NodeGene':
        return NodeGene(self)
    
    def __str__(self) -> str:
        return f"NodeGene {{type={self.type.name},innovationNumber = {self.innovation_number}}}"
